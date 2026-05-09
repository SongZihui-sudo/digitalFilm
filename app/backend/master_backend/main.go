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

	// 执行数据库迁移 (添加 is_admin 列)
	if err := app.db.MigrateAdminColumn(); err != nil {
		log.Printf("Warning: migration failed (may be ok if column exists): %v", err)
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

	router.POST("/api/auth/register", app.Register)
	router.POST("/api/auth/login", app.Login)
	router.POST("/api/auth/forgot-password", app.ForgotPassword)
	router.POST("/api/auth/reset-password", app.ResetPassword)

	// 管理员登录 (无需普通用户权限，独立接口)
	router.POST("/api/admin/login", app.AdminLogin)

	// 管理员专用路由 (需要管理员 JWT)
	adminGroup := router.Group("/api/admin")
	adminGroup.Use(AdminMiddleware())
	{
		adminGroup.GET("/users", app.AdminListUsers)
		adminGroup.POST("/users", app.AdminCreateUser)
		adminGroup.PUT("/users/:id/password", app.AdminChangePassword)
		adminGroup.DELETE("/users/:id", app.AdminDeleteUser)
		adminGroup.PUT("/users/:id/toggle-admin", app.AdminToggleAdmin)
	}

	// 需要登录的受保护路由
	protectedGroup := router.Group("/")
	protectedGroup.Use(AuthMiddleware()) // 应用 JWT 验证中间件
	{
		// 2. 里面全部使用 protectedGroup 来注册，不能再用 router！
		protectedGroup.GET("/api/auth/me", app.GetProfile)
		protectedGroup.POST("/api/create_projects", app.CreateProject)
		protectedGroup.GET("/api/projects", app.ProjectList)
		protectedGroup.POST("/internal/images/upload", app.UploadImage)
		protectedGroup.GET("/api/projects/:projectId/images", app.GetProjectImages)
		protectedGroup.POST("/internal/images/get", app.GetImageInfo)
		protectedGroup.POST("/internal/film/register_result", app.RegisterFilmResult)
		protectedGroup.GET("/api/film/presets", app.GetPresets)
		protectedGroup.GET("/api/images/:id/settings", app.GetImageEditSetting)
		protectedGroup.POST("/api/images/:id/settings", app.UpdateImageEditSetting)
		protectedGroup.DELETE("/api/projects/:id", app.DeleteProject)
		protectedGroup.DELETE("/api/images/:id", app.DeleteImage)
	}

	err = router.Run(fmt.Sprintf("%s:%s", backendConfig.Url, backendConfig.Port))
	if err != nil {
		log.Fatal(err)
	}
}
