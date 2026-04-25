package main

import (
	"backend/master_backend/db"
	"backend/utils"
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
)

type App struct {
	mu       sync.RWMutex
	projects []utils.Project
	images   map[string][]utils.ImageAsset
	db       *db.AppDb
	Cfg      utils.MasterBackendConfig
}

func NewApp(dbPath string, cfg utils.MasterBackendConfig) (*App, error) {
	appDB, err := db.NewAppDb(dbPath)
	if err != nil {
		return nil, fmt.Errorf("init db failed: %w", err)
	}

	if err := appDB.InitTables(); err != nil {
		return nil, fmt.Errorf("init tables failed: %w", err)
	}

	projects, err := appDB.LoadProjects()
	if err != nil {
		return nil, fmt.Errorf("load projects failed: %w", err)
	}

	images, err := appDB.LoadImages()
	if err != nil {
		return nil, fmt.Errorf("load images failed: %w", err)
	}

	app := &App{
		projects: projects,
		images:   images,
		db:       appDB,
		Cfg:      cfg,
	}

	return app, nil
}

func (app *App) setProjects(projces []utils.Project) {
	app.mu.Lock()
	app.projects = projces
	app.mu.Unlock()
}

func (app *App) GetProjects(ctx *gin.Context) {
	app.mu.Lock()
	defer app.mu.Unlock()
	ctx.JSON(http.StatusOK, gin.H{
		"ok":       true,
		"projects": app.projects,
	})
}

func (app *App) setImages(images map[string][]utils.ImageAsset) {
	app.mu.Lock()
	app.images = images
	app.mu.Unlock()
}

func (app *App) CloseDB() error {
	return app.db.CloseDB()
}

func (app *App) findOriginURLByImageID(imageID string) (string, error) {
	// 查数据库
	imageInfo, err := app.db.GetImageInfo(imageID)
	if err != nil {
		log.Fatal(err)
	}
	return imageInfo.OriginalURL, err
}

func (app *App) GetImageInfo(ctx *gin.Context) {
	var req GetImageRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"ok":    false,
			"error": err.Error(),
		})
		return
	}
	originURL, err := app.findOriginURLByImageID(req.ImageID)
	if err != nil {
		ctx.JSON(http.StatusNotFound, gin.H{
			"ok":    false,
			"error": "image not found",
		})
		return
	}
	ctx.JSON(http.StatusOK, GetImageResponse{
		OK:        true,
		ImageID:   req.ImageID,
		OriginURL: originURL,
	})
}
