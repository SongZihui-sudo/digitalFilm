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

	url := spew.Sprintf("%s:%s", config.Url, config.Port)
	log.Printf("server running at %s", url)

	if err := router.Run(url); err != nil {
		log.Fatal(err)
	}
}

func (app *App) uploadFileHandler(ctx *gin.Context) {
	saveUploadedFile(ctx, app.Cfg.UploadPath, "/uploads", app.Cfg.MainServerURL, fmt.Sprintf("http://%s:%s", app.Cfg.Url, app.Cfg.Port))
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

	// 转发给主服务器
	body, err := json.Marshal(img)
	if err != nil {
		_ = os.Remove(savePath)
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error": "marshal image metadata failed",
		})
		return
	}

	resp, err := http.Post(
		MainServerURL+"/internal/images/upload",
		"application/json",
		bytes.NewBuffer(body),
	)
	if err != nil {
		_ = os.Remove(savePath)
		ctx.JSON(http.StatusBadGateway, gin.H{
			"error": "forward to main server failed",
		})
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		_ = os.Remove(savePath)
		ctx.JSON(http.StatusBadGateway, gin.H{
			"error": "read main server response failed",
		})
		return
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_ = os.Remove(savePath)
		ctx.Data(resp.StatusCode, "application/json", respBody)
		return
	}

	// 直接把主服务器响应返回给前端
	ctx.Data(http.StatusOK, "application/json", respBody)
}
