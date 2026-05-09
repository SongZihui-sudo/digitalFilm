package main

import (
	"backend/utils"
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func LoadConfig(path string) (*AppConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file failed: %w", err)
	}
	var cfg AppConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config file failed: %w", err)
	}
	return &cfg, nil
}

type AppConfig struct {
	Port          string `json:"port"`
	Url           string `json:"url"`
	UploadPath    string `json:"UploadDir"`
	ResultPath    string `json:"ResultDir"`
	MainServerURL string `json:"MainServerURL"`
}

type App struct {
	Cfg AppConfig
}

func NewApp(cfg *AppConfig) *App {
	return &App{Cfg: *cfg}
}

func main() {
	config, err := LoadConfig("./config/static_backend.json")
	if err != nil {
		log.Fatal(err)
	}
	app := NewApp(config)

	if err := os.MkdirAll(config.UploadPath, 0755); err != nil {
		log.Fatal(err)
	}
	if err := os.MkdirAll(config.ResultPath, 0755); err != nil {
		log.Fatal(err)
	}

	router := gin.Default()

	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: false,
		MaxAge:           12 * time.Hour,
	}))

	router.Static("/uploads", config.UploadPath)
	router.Static("/results", config.ResultPath)

	router.GET("/ping", func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, gin.H{
			"message": "ok",
		})
	})

	router.POST("/api/images/upload", app.uploadFileHandler)
	router.DELETE("/api/images/:type/:id", app.deleteFileHandler)

	url := spew.Sprintf("%s:%s", config.Url, config.Port)
	log.Printf("server running at %s", url)

	if err := router.Run(url); err != nil {
		log.Fatal(err)
	}
}

func (app *App) uploadFileHandler(ctx *gin.Context) {
	saveUploadedFile(ctx, app.Cfg.UploadPath, "/uploads", app.Cfg.MainServerURL, fmt.Sprintf("http://%s:%s", app.Cfg.Url, app.Cfg.Port))
}

func (app *App) deleteFileHandler(ctx *gin.Context) {
	// 获取 URL 中的文件类型和文件名/ID (例如 source 或 result, 12345.jpg)
	fileType := ctx.Param("type")
	filename := ctx.Param("id")

	if filename == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "file id/name is required"})
		return
	}

	// 安全校验：防止目录穿越漏洞 (Directory Traversal)
	if strings.Contains(filename, "..") || strings.Contains(filename, "/") || strings.Contains(filename, "\\") {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "invalid filename"})
		return
	}

	// 根据文件类型选择文件的路径
	var filePath string
	if fileType == "source" {
		// 如果是源文件，使用 UploadPath
		filePath = filepath.Join(app.Cfg.UploadPath, filename)
	} else if fileType == "result" {
		// 如果是胶片结果文件，使用 ResultPath
		filePath = filepath.Join(app.Cfg.ResultPath, filename)
	} else {
		// 如果文件类型无效，返回错误
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "invalid file type"})
		return
	}

	// 执行删除
	err := os.Remove(filePath)
	if err != nil && !os.IsNotExist(err) { // 如果文件本身就不存在，直接放行即可
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error":   "failed to delete file on static server",
			"details": err.Error(),
		})
		return
	}

	// 删除成功
	ctx.JSON(http.StatusOK, gin.H{"message": "file deleted successfully"})
}

func saveUploadedFile(ctx *gin.Context, baseDir, urlPrefix string, MainServerURL string, PublicBaseURL string) {
	projectID := ctx.PostForm("project_id")
	if projectID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "project_id is required",
		})
		return
	}

	fileHeader, err := ctx.FormFile("file")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "file is required",
		})
		return
	}

	// 读图片尺寸
	src, err := fileHeader.Open()
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error": "open uploaded file failed",
		})
		return
	}
	defer src.Close()

	cfg, _, err := image.DecodeConfig(src)
	if err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid image file",
		})
		return
	}

	imageID := uuid.NewString()
	ext := filepath.Ext(fileHeader.Filename)
	if ext == "" {
		ext = ".bin"
	}

	filename := imageID + ext
	savePath := filepath.Join(baseDir, filename)

	if err := os.MkdirAll(baseDir, 0755); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error": "create upload dir failed",
		})
		return
	}

	if err := ctx.SaveUploadedFile(fileHeader, savePath); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error": "save file failed",
		})
		return
	}

	authHeader := ctx.GetHeader("Authorization")

	// 2. 准备转发给主服务器的数据
	img := utils.ImageAsset{
		ID:           imageID,
		ProjectID:    projectID,
		Name:         fileHeader.Filename,
		OriginalURL:  PublicBaseURL + urlPrefix + "/" + filename,
		ThumbnailURL: "",
		Width:        cfg.Width,
		Height:       cfg.Height,
		CreatedAt:    time.Now().Format(time.RFC3339),
	}

	body, err := json.Marshal(img)
	if err != nil {
		_ = os.Remove(savePath)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "marshal failed"})
		return
	}

	// 3. 使用 http.NewRequest 来构造请求，以便设置 Header
	req, err := http.NewRequest("POST", MainServerURL+"/internal/images/upload", bytes.NewBuffer(body))
	if err != nil {
		_ = os.Remove(savePath)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "create request failed"})
		return
	}

	// 重要：将原始用户的 Token 转发给 Main 服务器
	req.Header.Set("Content-Type", "application/json")
	if authHeader != "" {
		req.Header.Set("Authorization", authHeader)
	}

	// 4. 执行转发
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		_ = os.Remove(savePath)
		ctx.JSON(http.StatusBadGateway, gin.H{"error": "forward to main server failed"})
		return
	}
	defer resp.Body.Close()

	// ... 后续读取响应并返回给前端的逻辑保持不变 ...
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		_ = os.Remove(savePath)
		ctx.JSON(http.StatusBadGateway, gin.H{"error": "read main server response failed"})
		return
	}

	// 透传主服务器的状态码和内容
	ctx.Data(resp.StatusCode, "application/json", respBody)
}
