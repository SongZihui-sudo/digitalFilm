package main

import (
	"backend/utils"
	"net/http"
	"path"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/gin-gonic/gin"
)

type CreateProjectRequest struct {
	Name string `json:"name" binding:"required"`
}

func (app *App) CreateProject(ctx *gin.Context) {
	var req CreateProjectRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": err.Error(),
		})
		return
	}

	project := utils.Project{
		ID:        uuid.NewString(),
		Name:      req.Name,
		CreatedAt: time.Now().Format(time.RFC3339),
		CoverURL:  "",
	}

	if err := app.db.AppendProject(project); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	app.mu.Lock()
	app.projects = append(app.projects, project)
	app.mu.Unlock()

	ctx.JSON(http.StatusOK, project)
}

func (app *App) ProjectList(ctx *gin.Context) {
	app.mu.RLock()
	projects := make([]utils.Project, len(app.projects))
	copy(projects, app.projects)
	app.mu.RUnlock()

	ctx.JSON(http.StatusOK, projects)
}

func (app *App) UploadImage(ctx *gin.Context) {
	var img utils.ImageAsset
	if err := ctx.ShouldBindJSON(&img); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid image payload",
		})
		return
	}

	if img.ID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "image id is required",
		})
		return
	}
	if img.ProjectID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "projectId is required",
		})
		return
	}
	if img.Name == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "name is required",
		})
		return
	}
	if img.OriginalURL == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "originalUrl is required",
		})
		return
	}
	if img.CreatedAt == "" {
		img.CreatedAt = time.Now().Format(time.RFC3339)
	}

	// 检查 project 是否存在
	app.mu.RLock()
	projectExists := false
	for _, p := range app.projects {
		if p.ID == img.ProjectID {
			projectExists = true
			break
		}
	}
	app.mu.RUnlock()

	if !projectExists {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "project not found",
		})
		return
	}

	if err := app.db.AppendImage(img); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	app.mu.Lock()
	app.images[img.ProjectID] = append(app.images[img.ProjectID], img)
	app.mu.Unlock()

	ctx.JSON(http.StatusOK, gin.H{
		"message": "image created successfully",
		"image":   img,
	})
}

func (app *App) GetProjectImages(ctx *gin.Context) {
	projectID := ctx.Param("projectId")
	if projectID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "projectId is required",
		})
		return
	}

	images, err := app.db.GetProjectImages(projectID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	ctx.JSON(http.StatusOK, images)
}

type GetImageRequest struct {
	ImageID string `json:"image_id" binding:"required"`
}
type GetImageResponse struct {
	OK        bool   `json:"ok"`
	ImageID   string `json:"image_id"`
	OriginURL string `json:"origin_url"`
}
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

	// 将生成的胶片风格结果信息保存到数据库
	err := app.db.SaveFilmResult(req.ImageID, req.FileName, resultURL, req.Width, req.Height, req.Basic, req.Film, req.Device)
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

func (app *App) GetPresets(ctx *gin.Context) {
	ctx.JSON(http.StatusOK, app.Cfg.Presets)
}

func (app *App) GetImageEditSetting(ctx *gin.Context) {
	imageID := ctx.Param("id")
	if imageID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "image id is required",
		})
		return
	}

	settings, err := app.db.GetImageEditSettings(imageID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	ctx.JSON(http.StatusOK, settings)
}

func (app *App) UpdateImageEditSetting(ctx *gin.Context) {
	imageID := ctx.Param("id")
	if imageID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "image id is required",
		})
		return
	}

	var settings utils.ImageEditSettings
	if err := ctx.ShouldBindJSON(&settings); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
		})
		return
	}

	// 强制使用路径参数里的 id，避免前端 body 乱传
	settings.ImageID = imageID

	if err := app.db.SaveImageEditSettings(settings); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"message": "image edit settings updated successfully",
		"data":    settings,
	})
}

// DeleteProject API 路由处理函数
func (app *App) DeleteProject(ctx *gin.Context) {
	projectID := ctx.Param("id")

	// 校验 ID 是否为空
	if projectID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "project id is required"})
		return
	}

	// Delete the project from the database and static server
	err := app.db.DeleteProject(projectID, app.Cfg.StaticServer)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error":   "failed to delete project",
			"details": err.Error(),
		})
		return
	}

	// 重新加载
	projects, err := app.db.LoadProjects()
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error":   "failed to delete project",
			"details": err.Error(),
		})
		return
	}
	app.setProjects(projects)

	images, err := app.db.LoadImages()
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error":   "failed to delete project",
			"details": err.Error(),
		})
		return
	}
	app.setImages(images)

	// Return success response
	ctx.JSON(http.StatusOK, gin.H{
		"message":    "project deleted successfully",
		"project_id": projectID,
	})
}

// DeleteImage API 路由处理函数
func (app *App) DeleteImage(ctx *gin.Context) {
	imageID := ctx.Param("id")
	if imageID == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "image id is required"})
		return
	}

	// Delete the image from the database and static server
	err := app.db.DeleteImage(imageID, app.Cfg.StaticServer)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"error":   "failed to delete image",
			"details": err.Error(),
		})
		return
	}

	// Delete the image from in-memory data (images map)
	app.mu.Lock() // Acquire a write lock for thread-safety
	defer app.mu.Unlock()

	// Find and remove the image from the memory
	for projectID, images := range app.images {
		for i, img := range images {
			if img.ID == imageID {
				// Remove the image from the in-memory list for this project
				app.images[projectID] = append(images[:i], images[i+1:]...)
				break
			}
		}
	}

	// Return success response
	ctx.JSON(http.StatusOK, gin.H{
		"message": "image deleted successfully",
		"id":      imageID,
	})
}
