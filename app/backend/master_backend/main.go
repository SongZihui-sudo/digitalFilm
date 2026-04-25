package main

import (
	"backend/utils"
	"fmt"
	"log"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

var ConfigPath string = "./config/master_backend.json"

func main() {
	backendConfig, err := utils.LoadConfig(ConfigPath)
	if err != nil {
		log.Fatal(err)
	}

	app, err := NewApp(backendConfig.DBPATH, *backendConfig)
	defer func(app *App) {
		err := app.CloseDB()
		if err != nil {
			log.Fatal(err)
		}
	}(app)

	if err != nil {
		panic(err)
	}

	router := gin.Default()
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	router.POST("/api/create_projects", app.CreateProject)
	router.GET("/api/projects", app.ProjectList)
	router.POST("/internal/images/upload", app.UploadImage)
	router.GET("/api/projects/:projectId/images", app.GetProjectImages)
	router.POST("/internal/images/get", app.GetImageInfo)
	router.POST("/internal/film/register_result", app.RegisterFilmResult)
	router.GET("/api/film/presets", app.GetPresets)
	router.GET("/api/images/:id/settings", app.GetImageEditSetting)
	router.POST("/api/images/:id/settings", app.UpdateImageEditSetting)
	router.DELETE("/api/projects/:id", app.DeleteProject)
	router.DELETE("/api/images/:id", app.DeleteImage)

	err = router.Run(fmt.Sprintf("%s:%s", backendConfig.Url, backendConfig.Port))
	if err != nil {
		log.Fatal(err)
	}
}
