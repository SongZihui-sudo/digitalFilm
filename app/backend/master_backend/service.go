package main

import (
	"backend/utils"
	"net/http"
	"path"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// ==========================================
// 1. 项目相关接口
// ==========================================

type CreateProjectRequest struct {
	Name string `json:"name" binding:"required"`
}

func (app *App) CreateProject(ctx *gin.Context) {
	// 从 JWT 中间件获取当前用户 ID
	userID := ctx.GetString("userID")

	var req CreateProjectRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	project := utils.Project{
		ID:        uuid.NewString(),
		UserID:    userID, // 绑定用户
		Name:      req.Name,
		CreatedAt: time.Now().Format(time.RFC3339),
		CoverURL:  "",
	}

	if err := app.db.AppendProject(project); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// 移除缓存锁，直接返回
	ctx.JSON(http.StatusOK, project)
}

func (app *App) ProjectList(ctx *gin.Context) {
	userID := ctx.GetString("userID")

	// 直接从数据库读取属于该用户的项目列表
	projects, err := app.db.LoadProjects(userID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// 如果没有项目，返回空数组而不是 null
	if projects == nil {
		projects = []utils.Project{}
	}

	ctx.JSON(http.StatusOK, projects)
}

func (app *App) DeleteProject(ctx *gin.Context) {
	userID := ctx.GetString("userID")
	projectID := ctx.Param("id")

	if projectID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "project id is required"})
		return
	}

	// 传入 userID 进行越权检查并删除
	err := app.db.DeleteProject(projectID, userID, app.Cfg.StaticServer)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error":   "failed to delete project",
			"details": err.Error(),
		})
		return
	}

	// 移除了重新加载到内存的逻辑

	ctx.JSON(http.StatusOK, gin.H{
		"message":    "project deleted successfully",
		"project_id": projectID,
	})
}

// ==========================================
// 2. 图片相关接口
// ==========================================

func (app *App) UploadImage(ctx *gin.Context) {
	userID := ctx.GetString("userID")

	var img utils.ImageAsset
	if err := ctx.ShouldBindJSON(&img); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "invalid image payload"})
		return
	}

	if img.ID == "" || img.ProjectID == "" || img.Name == "" || img.OriginalURL == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "missing required fields"})
		return
	}

	if img.CreatedAt == "" {
		img.CreatedAt = time.Now().Format(time.RFC3339)
	}

	img.UserID = userID // 强制绑定当前用户

	// 检查 project 是否存在且属于当前用户
	projects, err := app.db.LoadProjects(userID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "failed to verify project"})
		return
	}

	projectExists := false
	for _, p := range projects {
		if p.ID == img.ProjectID {
			projectExists = true
			break
		}
	}

	if !projectExists {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "project not found or permission denied"})
		return
	}

	// 插入数据库
	if err := app.db.AppendImage(img); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"message": "image created successfully",
		"image":   img,
	})
}

func (app *App) GetProjectImages(ctx *gin.Context) {
	userID := ctx.GetString("userID")
	projectID := ctx.Param("projectId")

	if projectID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "projectId is required"})
		return
	}

	// 传入 userID 获取数据
	images, err := app.db.GetProjectImages(projectID, userID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if images == nil {
		images = []utils.ImageAsset{}
	}

	ctx.JSON(http.StatusOK, images)
}

func (app *App) DeleteImage(ctx *gin.Context) {
	userID := ctx.GetString("userID")
	imageID := ctx.Param("id")

	if imageID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "image id is required"})
		return
	}

	// 传入 userID 进行安全校验删除
	err := app.db.DeleteImage(imageID, userID, app.Cfg.StaticServer)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error":   "failed to delete image",
			"details": err.Error(),
		})
		return
	}

	// 移除了操作内存切片的逻辑

	ctx.JSON(http.StatusOK, gin.H{
		"message": "image deleted successfully",
		"id":      imageID,
	})
}

// ==========================================
// 3. 胶片预设与生成相关接口
// ==========================================

func (app *App) GetPresets(ctx *gin.Context) {
	// 预设列表一般是全局的，可以直接返回
	ctx.JSON(http.StatusOK, app.Cfg.Presets)
}

func (app *App) GetImageEditSetting(ctx *gin.Context) {
	userID := ctx.GetString("userID")
	imageID := ctx.Param("id")

	if imageID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "image id is required"})
		return
	}

	// 为安全起见，先校验该图片是否属于当前用户
	var checkID string
	err := app.db.DB.QueryRow("SELECT id FROM image_assets WHERE id = ? AND user_id = ?", imageID, userID).Scan(&checkID)
	if err != nil {
		ctx.JSON(http.StatusForbidden, gin.H{"error": "permission denied or image not found"})
		return
	}

	settings, err := app.db.GetImageEditSettings(imageID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, settings)
}

func (app *App) UpdateImageEditSetting(ctx *gin.Context) {
	userID := ctx.GetString("userID")
	imageID := ctx.Param("id")

	if imageID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "image id is required"})
		return
	}

	var settings utils.ImageEditSettings
	if err := ctx.ShouldBindJSON(&settings); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	// 强制使用路径参数里的 id，并绑定当前请求的 userID
	settings.ImageID = imageID
	settings.UserID = userID

	if err := app.db.SaveImageEditSettings(settings); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"message": "image edit settings updated successfully",
		"data":    settings,
	})
}

// ==========================================
// 4. 胶片结果注册
// ==========================================

type RegisterResultRequest struct {
	ImageID      string                 `json:"image_id" binding:"required"`
	FileName     string                 `json:"file_name" binding:"required"`
	RelativePath string                 `json:"relative_path" binding:"required"`
	Width        int                    `json:"width"`
	Height       int                    `json:"height"`
	Basic        map[string]interface{} `json:"basic"`
	Film         map[string]interface{} `json:"film"`
	Device       string                 `json:"device"`
}

type RegisterResultResponse struct {
	OK        bool   `json:"ok"`
	ResultURL string `json:"result_url"`
}

func (app *App) RegisterFilmResult(ctx *gin.Context) {
	userID := ctx.GetString("userID")

	var req RegisterResultRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"ok":    false,
			"error": err.Error(),
		})
		return
	}

	relative := strings.TrimLeft(req.RelativePath, "/")
	resultURL := app.Cfg.StaticServer + "/" + path.Clean(relative)

	// 传入 userID
	err := app.db.SaveFilmResult(userID, req.ImageID, req.FileName, resultURL, req.Width, req.Height, req.Basic, req.Film, req.Device)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"ok":    false,
			"error": err.Error(),
		})
		return
	}

	ctx.JSON(http.StatusOK, RegisterResultResponse{
		OK:        true,
		ResultURL: resultURL,
	})
}
