package main

import (
	"backend/master_backend/db"
	"backend/utils"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

// 定义一个 JWT 签名的密钥（实际部署时应放在配置文件 Cfg 中）
var jwtSecret = []byte("your-super-secret-key")

type App struct {
	db  *db.AppDb
	Cfg utils.MasterBackendConfig
}

func NewApp(dbPath string, cfg utils.MasterBackendConfig) (*App, error) {
	appDB, err := db.NewAppDb(dbPath)
	if err != nil {
		return nil, fmt.Errorf("init db failed: %w", err)
	}

	if err := appDB.InitTables(); err != nil {
		return nil, fmt.Errorf("init tables failed: %w", err)
	}

	// 移除全局 LoadProjects 和 LoadImages 逻辑，因为数据现在归属不同用户

	app := &App{
		db:  appDB,
		Cfg: cfg,
	}

	return app, nil
}

func (app *App) CloseDB() error {
	return app.db.CloseDB()
}

// RegisterReq 注册请求体
type RegisterReq struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required,min=6"`
	Email    string `json:"email"`
}

func (app *App) Register(ctx *gin.Context) {
	var req RegisterReq
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"ok": false, "error": err.Error()})
		return
	}

	user, err := app.db.RegisterUser(req.Username, req.Password, req.Email)
	if err != nil {
		ctx.JSON(http.StatusConflict, gin.H{"ok": false, "error": "用户名可能已存在或注册失败"})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"ok":   true,
		"user": user,
	})
}

// LoginReq 登录请求体
type LoginReq struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

func (app *App) Login(ctx *gin.Context) {
	var req LoginReq
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"ok": false, "error": err.Error()})
		return
	}

	user, err := app.db.LoginUser(req.Username, req.Password)
	if err != nil {
		ctx.JSON(http.StatusUnauthorized, gin.H{"ok": false, "error": "账号或密码错误"})
		return
	}

	// 生成 JWT Token
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"user_id": user.ID,
		"exp":     time.Now().Add(24 * time.Hour * 7).Unix(), // 7天过期
	})

	tokenString, err := token.SignedString(jwtSecret)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"ok": false, "error": "生成 Token 失败"})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"ok":    true,
		"token": tokenString,
		"user":  user,
	})
}

// AuthMiddleware 验证 JWT 并将 userID 存入 context
func AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		tokenString := c.GetHeader("Authorization")
		// 通常前端会在 header 里加上: Authorization: Bearer <token>
		if len(tokenString) > 7 && tokenString[:7] == "Bearer " {
			tokenString = tokenString[7:]
		}

		if tokenString == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"ok": false, "error": "未提供授权 token"})
			return
		}

		token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}
			return jwtSecret, nil
		})

		if err != nil || !token.Valid {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"ok": false, "error": "token 无效或已过期"})
			return
		}

		claims, ok := token.Claims.(jwt.MapClaims)
		if !ok {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"ok": false, "error": "无法解析 token 负载"})
			return
		}

		userID, ok := claims["user_id"].(string)
		if !ok {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"ok": false, "error": "token 格式错误"})
			return
		}

		// 将用户 ID 挂载到上下文中，后续的 Handler 都可以直接获取
		c.Set("userID", userID)
		c.Next()
	}
}

func (app *App) GetProfile(ctx *gin.Context) {
	userID := ctx.GetString("userID")

	user, err := app.db.GetUserInfo(userID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"ok": false, "error": "获取用户信息失败"})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{"ok": true, "user": user})
}

func (app *App) GetProjects(ctx *gin.Context) {
	// 从中间件获取当前登录的用户 ID
	userID := ctx.GetString("userID")

	// 直接从数据库中加载属于该用户的项目
	projects, err := app.db.LoadProjects(userID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
			"ok":    false,
			"error": "加载项目列表失败",
		})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"ok":       true,
		"projects": projects,
	})
}

// 对应你在问题中给出的接口
func (app *App) GetImageInfo(ctx *gin.Context) {
	var req struct {
		ImageID string `json:"image_id" binding:"required"`
	}
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"ok": false, "error": err.Error()})
		return
	}

	// 这里需要注意：你应该在数据库层面的 GetImageInfo 里加上 user_id 限制
	// 防止用户拿到别人的 image_id 来请求数据
	userID := ctx.GetString("userID")

	// 我们直接利用数据库查询来验证归属权并获取信息
	var originURL string
	err := app.db.DB.QueryRow(`
		SELECT original_url 
		FROM image_assets 
		WHERE id = ? AND user_id = ?
	`, req.ImageID, userID).Scan(&originURL)

	if err != nil {
		ctx.JSON(http.StatusNotFound, gin.H{
			"ok":    false,
			"error": "图片不存在或无权访问",
		})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"ok":         true,
		"image_id":   req.ImageID,
		"origin_url": originURL,
	})
}

type ForgotPasswordReq struct {
	Email string `json:"email" binding:"required,email"`
}

// ForgotPassword 处理忘记密码请求 (发送重置邮件)
func (app *App) ForgotPassword(ctx *gin.Context) {
	var req ForgotPasswordReq
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "无效的邮箱格式"})
		return
	}

	// 1. 检查邮箱是否存在于数据库
	exists, err := app.db.CheckEmailExists(req.Email)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "数据库查询失败"})
		return
	}

	if !exists {
		// 出于安全考虑（防止别人枚举邮箱），通常即使邮箱不存在也返回成功提示，
		// 但这里为了方便你调试，明确返回错误。
		ctx.JSON(http.StatusNotFound, gin.H{"error": "该邮箱尚未注册"})
		return
	}

	// 2. 核心逻辑：生成 Reset Token 并发送邮件 (这里用伪代码代替真实邮件服务)
	/*
		resetToken := generateSecureToken()
		saveTokenToDBOrRedis(req.Email, resetToken, 15*time.Minute)
		sendEmail(req.Email, "您的重置链接：https://yoursite.com/reset?token=" + resetToken)
	*/
	fmt.Printf("[模拟发送邮件] 您的重置申请已受理，邮箱: %s\n", req.Email)

	// 告诉前端处理成功
	ctx.JSON(http.StatusOK, gin.H{
		"ok":      true,
		"message": "重置指引已发送至您的邮箱",
	})
}

// =======================================================
// 下面是用户收到邮件后，点击链接设置新密码的接口
// =======================================================

type ResetPasswordReq struct {
	Email string `json:"email" binding:"required,email"`
	// Token    string `json:"token" binding:"required"` // 实际生产中必须校验 Token
	NewPassword string `json:"new_password" binding:"required,min=6"`
}

// ResetPassword 执行重置密码操作
func (app *App) ResetPassword(ctx *gin.Context) {
	var req ResetPasswordReq
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求参数"})
		return
	}

	// 实际生产中：必须在这里校验 Token 是否合法、是否过期
	// if !verifyResetToken(req.Email, req.Token) {
	//     ctx.JSON(http.StatusUnauthorized, gin.H{"error": "重置链接已失效或不合法"})
	//     return
	// }

	// 调用底层数据库方法强制重置密码
	err := app.db.ResetPasswordByEmail(req.Email, req.NewPassword)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "重置密码失败"})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"ok":      true,
		"message": "密码重置成功，请使用新密码登录",
	})
}
