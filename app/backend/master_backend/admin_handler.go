package main

import (
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

// AdminMiddleware 验证 JWT 并将 userID 存入 context (管理员专用)
// 权限校验已在 AdminLogin 中完成，中间件只验证 JWT 合法性
func AdminMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		tokenString := c.GetHeader("Authorization")
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

		c.Set("userID", userID)
		c.Next()
	}
}

// AdminLogin 管理员登录接口
func (app *App) AdminLogin(ctx *gin.Context) {
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

	if !user.IsAdmin {
		ctx.JSON(http.StatusForbidden, gin.H{"ok": false, "error": "无管理员权限"})
		return
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"user_id":  user.ID,
		"is_admin": true,
		"exp":      time.Now().Add(24 * time.Hour).Unix(),
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

// AdminListUsers 管理员查看所有用户列表
func (app *App) AdminListUsers(ctx *gin.Context) {
	users, err := app.db.ListAllUsers()
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"ok": false, "error": "获取用户列表失败"})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{"ok": true, "users": users})
}

type AdminChangePasswordReq struct {
	NewPassword string `json:"new_password" binding:"required,min=6"`
}

// AdminChangePassword 管理员修改用户密码
func (app *App) AdminChangePassword(ctx *gin.Context) {
	userID := ctx.Param("id")

	var req AdminChangePasswordReq
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"ok": false, "error": "密码至少需要6位"})
		return
	}

	if err := app.db.AdminSetPassword(userID, req.NewPassword); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"ok": false, "error": "修改密码失败: " + err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{"ok": true, "message": "密码修改成功"})
}

type AdminCreateUserReq struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required,min=6"`
	Email    string `json:"email"`
}

// AdminCreateUser 管理员创建新用户
func (app *App) AdminCreateUser(ctx *gin.Context) {
	var req AdminCreateUserReq
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"ok": false, "error": err.Error()})
		return
	}

	user, err := app.db.AdminCreateUser(req.Username, req.Password, req.Email)
	if err != nil {
		ctx.JSON(http.StatusConflict, gin.H{"ok": false, "error": "用户名可能已存在或创建失败"})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{"ok": true, "user": user})
}

// AdminDeleteUser 管理员删除用户
func (app *App) AdminDeleteUser(ctx *gin.Context) {
	userID := ctx.Param("id")

	if err := app.db.AdminDeleteUser(userID); err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"ok": false, "error": "删除用户失败: " + err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{"ok": true, "message": "用户已删除"})
}

// AdminToggleAdmin 切换用户管理员权限
func (app *App) AdminToggleAdmin(ctx *gin.Context) {
	userID := ctx.Param("id")

	user, err := app.db.AdminToggleAdmin(userID)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"ok": false, "error": "操作失败: " + err.Error()})
		return
	}

	ctx.JSON(http.StatusOK, gin.H{"ok": true, "user": user})
}
