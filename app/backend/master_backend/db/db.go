package db

import (
	"backend/utils"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"golang.org/x/crypto/bcrypt"
	_ "modernc.org/sqlite"
)

const (
	CLOSED int = iota
	RUNNING
	ERROR
)

type AppDb struct {
	DB     *sql.DB
	Status int
}

func NewAppDb(dbPath string) (*AppDb, error) {
	curDB := &AppDb{
		Status: CLOSED,
	}

	if err := curDB.OpenDB(dbPath); err != nil {
		return nil, err
	}

	return curDB, nil
}

func (DBPtr *AppDb) OpenDB(path string) error {
	dbConn, err := sql.Open("sqlite", path)
	if err != nil {
		DBPtr.Status = ERROR
		return fmt.Errorf("open sqlite failed: %w", err)
	}

	// 开启外键约束支持
	if _, err := dbConn.Exec("PRAGMA foreign_keys = ON;"); err != nil {
		return fmt.Errorf("enable foreign keys failed: %w", err)
	}

	if err := dbConn.Ping(); err != nil {
		DBPtr.Status = ERROR
		return fmt.Errorf("ping sqlite failed: %w", err)
	}

	DBPtr.DB = dbConn
	DBPtr.Status = RUNNING
	return nil
}

func (DBPtr *AppDb) CloseDB() error {
	err := DBPtr.DB.Close()
	DBPtr.Status = CLOSED
	return err
}

func (DBPtr *AppDb) InitTables() error {
	// 用户表 (新增)
	userTableSQL := `
	CREATE TABLE IF NOT EXISTS users (
		id TEXT PRIMARY KEY,
		username TEXT UNIQUE NOT NULL,
		password_hash TEXT NOT NULL,
		email TEXT DEFAULT '',
		created_at TEXT NOT NULL
	);
	`

	// 项目表 (加入 user_id)
	projectTableSQL := `
	CREATE TABLE IF NOT EXISTS projects (
		id TEXT PRIMARY KEY,
		user_id TEXT NOT NULL,
		name TEXT NOT NULL,
		created_at TEXT NOT NULL,
		cover_url TEXT DEFAULT '',
		FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
	);
	`

	// 图片表 (加入 user_id)
	imageTableSQL := `
	CREATE TABLE IF NOT EXISTS image_assets (
		id TEXT PRIMARY KEY,
		user_id TEXT NOT NULL,
		project_id TEXT NOT NULL,
		name TEXT NOT NULL,
		original_url TEXT NOT NULL,
		thumbnail_url TEXT DEFAULT '',
		width INTEGER DEFAULT 0,
		height INTEGER DEFAULT 0,
		created_at TEXT NOT NULL,
		FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
		FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
	);
	`

	// 图片编辑设置表 (加入 user_id)
	imageSettingsTableSQL := `
	CREATE TABLE IF NOT EXISTS image_edit_settings (
		image_id TEXT PRIMARY KEY,
		user_id TEXT NOT NULL,
		exposure INTEGER DEFAULT 0,
		contrast INTEGER DEFAULT 0,
		highlights INTEGER DEFAULT 0,
		shadows INTEGER DEFAULT 0,
		temperature INTEGER DEFAULT 0,
		tint INTEGER DEFAULT 0,
		saturation INTEGER DEFAULT 0,
		preset TEXT DEFAULT '',
		grain INTEGER DEFAULT 0,
		halation INTEGER DEFAULT 0,
		FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
		FOREIGN KEY(image_id) REFERENCES image_assets(id) ON DELETE CASCADE
	);
	`

	// 胶片风格结果表 (加入 user_id)
	filmResultTableSQL := `
	CREATE TABLE IF NOT EXISTS film_results (
		id TEXT PRIMARY KEY,
		user_id TEXT NOT NULL,
		image_id TEXT NOT NULL,
		file_name TEXT NOT NULL,
		result_url TEXT NOT NULL,
		width INTEGER DEFAULT 0,
		height INTEGER DEFAULT 0,
		basic_info TEXT DEFAULT '',
		film_info TEXT DEFAULT '',
		device TEXT DEFAULT '',
		created_at TEXT NOT NULL,
		FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
		FOREIGN KEY(image_id) REFERENCES image_assets(id) ON DELETE CASCADE
	);
	`

	// 依次创建各表
	tables := []string{userTableSQL, projectTableSQL, imageTableSQL, imageSettingsTableSQL, filmResultTableSQL}
	for _, sqlStmt := range tables {
		if _, err := DBPtr.DB.Exec(sqlStmt); err != nil {
			DBPtr.Status = ERROR
			return fmt.Errorf("init table failed: %w", err)
		}
	}

	return nil
}

// ==========================================
// 用户认证相关方法
// ==========================================

// RegisterUser 注册新用户
func (DBPtr *AppDb) RegisterUser(username, password, email string) (*utils.User, error) {
	// 密码哈希
	hashedBytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, fmt.Errorf("hash password failed: %w", err)
	}

	user := &utils.User{
		ID:        utils.GenerateUniqueID(),
		Username:  username,
		Email:     email,
		CreatedAt: time.Now().Format(time.RFC3339),
	}

	_, err = DBPtr.DB.Exec(`
		INSERT INTO users (id, username, password_hash, email, created_at)
		VALUES (?, ?, ?, ?, ?)
	`, user.ID, user.Username, string(hashedBytes), user.Email, user.CreatedAt)

	if err != nil {
		return nil, fmt.Errorf("insert user failed (username might exist): %w", err)
	}

	return user, nil
}

// LoginUser 用户登录验证
func (DBPtr *AppDb) LoginUser(username, password string) (*utils.User, error) {
	var user utils.User
	var passwordHash string

	err := DBPtr.DB.QueryRow(`
		SELECT id, username, password_hash, email, created_at
		FROM users WHERE username = ?
	`, username).Scan(&user.ID, &user.Username, &passwordHash, &user.Email, &user.CreatedAt)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("user not found")
		}
		return nil, fmt.Errorf("query user failed: %w", err)
	}

	// 验证密码
	err = bcrypt.CompareHashAndPassword([]byte(passwordHash), []byte(password))
	if err != nil {
		return nil, fmt.Errorf("invalid password")
	}

	return &user, nil
}

// GetUserInfo 获取用户信息
func (DBPtr *AppDb) GetUserInfo(userID string) (*utils.User, error) {
	var user utils.User

	err := DBPtr.DB.QueryRow(`
		SELECT id, username, email, created_at
		FROM users WHERE id = ?
	`, userID).Scan(&user.ID, &user.Username, &user.Email, &user.CreatedAt)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("user not found")
		}
		return nil, fmt.Errorf("query user info failed: %w", err)
	}

	return &user, nil
}

// ChangePassword 修改密码
func (DBPtr *AppDb) ChangePassword(userID, oldPassword, newPassword string) error {
	var passwordHash string
	err := DBPtr.DB.QueryRow("SELECT password_hash FROM users WHERE id = ?", userID).Scan(&passwordHash)
	if err != nil {
		return fmt.Errorf("fetch user failed: %w", err)
	}

	// 验证旧密码
	if err := bcrypt.CompareHashAndPassword([]byte(passwordHash), []byte(oldPassword)); err != nil {
		return fmt.Errorf("invalid old password")
	}

	// 生成新密码哈希
	newHashedBytes, err := bcrypt.GenerateFromPassword([]byte(newPassword), bcrypt.DefaultCost)
	if err != nil {
		return fmt.Errorf("hash new password failed: %w", err)
	}

	_, err = DBPtr.DB.Exec("UPDATE users SET password_hash = ? WHERE id = ?", string(newHashedBytes), userID)
	if err != nil {
		return fmt.Errorf("update password failed: %w", err)
	}

	return nil
}

// ResetPasswordByEmail 忘记密码专用：强制重置密码（不需要验证旧密码）
func (DBPtr *AppDb) ResetPasswordByEmail(email, newPassword string) error {
	// 生成新密码哈希
	newHashedBytes, err := bcrypt.GenerateFromPassword([]byte(newPassword), bcrypt.DefaultCost)
	if err != nil {
		return fmt.Errorf("hash new password failed: %w", err)
	}

	// 根据 email 更新密码
	result, err := DBPtr.DB.Exec("UPDATE users SET password_hash = ? WHERE email = ?", string(newHashedBytes), email)
	if err != nil {
		return fmt.Errorf("update password failed: %w", err)
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		// 如果影响的行数为 0，说明该邮箱没有注册
		return fmt.Errorf("user not found with this email")
	}

	return nil
}

// CheckEmailExists 检查邮箱是否已注册 (用于发送重置邮件前)
func (DBPtr *AppDb) CheckEmailExists(email string) (bool, error) {
	var id string
	err := DBPtr.DB.QueryRow("SELECT id FROM users WHERE email = ?", email).Scan(&id)
	if err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// ==========================================
// 业务数据相关方法 (均加入 userID 隔离)
// ==========================================

func (DBPtr *AppDb) LoadProjects(userID string) ([]utils.Project, error) {
	rows, err := DBPtr.DB.Query(`
		SELECT id, user_id, name, created_at, cover_url
		FROM projects
		WHERE user_id = ?
		ORDER BY created_at DESC
	`, userID)
	if err != nil {
		return nil, fmt.Errorf("query projects failed: %w", err)
	}
	defer rows.Close()

	var projects []utils.Project
	for rows.Next() {
		var p utils.Project
		if err := rows.Scan(&p.ID, &p.UserID, &p.Name, &p.CreatedAt, &p.CoverURL); err != nil {
			return nil, fmt.Errorf("scan project failed: %w", err)
		}
		projects = append(projects, p)
	}

	return projects, nil
}

func (DBPtr *AppDb) AppendProject(project utils.Project) error {
	_, err := DBPtr.DB.Exec(`
		INSERT INTO projects (id, user_id, name, created_at, cover_url)
		VALUES (?, ?, ?, ?, ?)
	`, project.ID, project.UserID, project.Name, project.CreatedAt, project.CoverURL)
	if err != nil {
		return fmt.Errorf("append project failed: %w", err)
	}
	return nil
}

func (DBPtr *AppDb) AppendImage(image utils.ImageAsset) error {
	_, err := DBPtr.DB.Exec(`
		INSERT INTO image_assets (
			id, user_id, project_id, name, original_url, thumbnail_url, width, height, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`,
		image.ID, image.UserID, image.ProjectID, image.Name, image.OriginalURL,
		image.ThumbnailURL, image.Width, image.Height, image.CreatedAt,
	)
	if err != nil {
		return fmt.Errorf("append image failed: %w", err)
	}
	return nil
}

func (DBPtr *AppDb) GetProjectImages(projectID, userID string) ([]utils.ImageAsset, error) {
	rows, err := DBPtr.DB.Query(`
		SELECT id, user_id, project_id, name, original_url, thumbnail_url, width, height, created_at
		FROM image_assets
		WHERE project_id = ? AND user_id = ?
		ORDER BY created_at DESC
	`, projectID, userID)
	if err != nil {
		return nil, fmt.Errorf("query project images failed: %w", err)
	}
	defer rows.Close()

	var images []utils.ImageAsset
	for rows.Next() {
		var img utils.ImageAsset
		if err := rows.Scan(
			&img.ID, &img.UserID, &img.ProjectID, &img.Name, &img.OriginalURL,
			&img.ThumbnailURL, &img.Width, &img.Height, &img.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan image_asset failed: %w", err)
		}
		images = append(images, img)
	}

	return images, nil
}

func (DBPtr *AppDb) GetImageEditSettings(imageID string) (utils.ImageEditSettings, error) {
	var settings utils.ImageEditSettings

	err := DBPtr.DB.QueryRow(`
		SELECT 
			image_id,
			user_id,
			exposure,
			contrast,
			highlights,
			shadows,
			temperature,
			tint,
			saturation,
			preset,
			grain,
			halation
		FROM image_edit_settings
		WHERE image_id = ?
	`, imageID).Scan(
		&settings.ImageID,
		&settings.UserID,
		&settings.Exposure,
		&settings.Contrast,
		&settings.Highlights,
		&settings.Shadows,
		&settings.Temperature,
		&settings.Tint,
		&settings.Saturation,
		&settings.Preset,
		&settings.Grain,
		&settings.Halation,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			// 如果数据库中还没有该图片的设置，返回空的结构体和一个明确的错误提示
			return utils.ImageEditSettings{}, fmt.Errorf("image edit settings not found: %s", imageID)
		}
		// 其他数据库查询错误
		return utils.ImageEditSettings{}, fmt.Errorf("get image edit settings failed: %w", err)
	}

	return settings, nil
}

func (DBPtr *AppDb) SaveImageEditSettings(settings utils.ImageEditSettings) error {
	_, err := DBPtr.DB.Exec(`
		INSERT INTO image_edit_settings (
			image_id, user_id, exposure, contrast, highlights, shadows, 
			temperature, tint, saturation, preset, grain, halation
		)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(image_id) DO UPDATE SET
			exposure = excluded.exposure,
			contrast = excluded.contrast,
			highlights = excluded.highlights,
			shadows = excluded.shadows,
			temperature = excluded.temperature,
			tint = excluded.tint,
			saturation = excluded.saturation,
			preset = excluded.preset,
			grain = excluded.grain,
			halation = excluded.halation
	`,
		settings.ImageID, settings.UserID, settings.Exposure, settings.Contrast, settings.Highlights,
		settings.Shadows, settings.Temperature, settings.Tint, settings.Saturation,
		settings.Preset, settings.Grain, settings.Halation,
	)
	if err != nil {
		return fmt.Errorf("save image edit settings failed: %w", err)
	}
	return nil
}

func (DBPtr *AppDb) DeleteImage(id, userID, staticServer string) error {
	// 防越权：确保该图片确实属于该用户
	var checkID string
	err := DBPtr.DB.QueryRow("SELECT id FROM image_assets WHERE id = ? AND user_id = ?", id, userID).Scan(&checkID)
	if err != nil {
		if err == sql.ErrNoRows {
			return fmt.Errorf("permission denied or image not found")
		}
		return fmt.Errorf("verify image ownership failed: %w", err)
	}

	// 1. 查找胶片风格结果的文件名
	var filmFileName string
	err = DBPtr.DB.QueryRow("SELECT file_name FROM film_results WHERE image_id = ?", id).Scan(&filmFileName)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("failed to fetch film result file name: %w", err)
	}

	// 2. 删除服务器静态文件
	if err == nil && filmFileName != "" {
		if err := utils.DeleteFileFromStaticServer("result", filmFileName, staticServer); err != nil {
			return fmt.Errorf("failed to delete film result file: %w", err)
		}
	}
	if err := utils.DeleteFileFromStaticServer("source", id, staticServer); err != nil {
		return fmt.Errorf("failed to delete image file: %w", err)
	}

	// 3. 数据库删除
	// 因为在建表时加上了 ON DELETE CASCADE，理论上可以直接删除 image_assets，级联表会自动清除
	// 但为了兼容，我们依然显式使用事务处理
	tx, err := DBPtr.DB.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	defer func() {
		if err != nil {
			tx.Rollback()
		} else {
			err = tx.Commit()
		}
	}()

	if _, err = tx.Exec("DELETE FROM film_results WHERE image_id = ?", id); err != nil {
		return fmt.Errorf("delete film result failed: %w", err)
	}
	if _, err = tx.Exec("DELETE FROM image_edit_settings WHERE image_id = ?", id); err != nil {
		return fmt.Errorf("delete image settings failed: %w", err)
	}
	if _, err = tx.Exec("DELETE FROM image_assets WHERE id = ?", id); err != nil {
		return fmt.Errorf("delete image asset failed: %w", err)
	}

	return nil
}

func (DBPtr *AppDb) DeleteProject(id, userID, staticServer string) error {
	// 验证项目归属权
	var checkID string
	err := DBPtr.DB.QueryRow("SELECT id FROM projects WHERE id = ? AND user_id = ?", id, userID).Scan(&checkID)
	if err != nil {
		if err == sql.ErrNoRows {
			return fmt.Errorf("permission denied or project not found")
		}
		return fmt.Errorf("verify project ownership failed: %w", err)
	}

	rows, err := DBPtr.DB.Query("SELECT id FROM image_assets WHERE project_id = ?", id)
	if err != nil {
		return fmt.Errorf("failed to query images: %w", err)
	}

	var imageIDs []string
	for rows.Next() {
		var imgID string
		if err := rows.Scan(&imgID); err != nil {
			rows.Close()
			return fmt.Errorf("failed to scan image id: %w", err)
		}
		imageIDs = append(imageIDs, imgID)
	}
	rows.Close()

	// 逐个删除图片及文件 (重用上面的 DeleteImage 函数，已包含防越权验证)
	for _, imgID := range imageIDs {
		if err := DBPtr.DeleteImage(imgID, userID, staticServer); err != nil {
			return fmt.Errorf("failed to delete image %s: %w", imgID, err)
		}
	}

	// 删除项目记录（因为有关联 ON DELETE CASCADE，其他数据库级联数据已被清除）
	if _, err := DBPtr.DB.Exec("DELETE FROM projects WHERE id = ?", id); err != nil {
		return fmt.Errorf("failed to delete project: %w", err)
	}

	return nil
}

func (DBPtr *AppDb) SaveFilmResult(userID, imageID, fileName, resultURL string, width, height int, basicInfo, filmInfo map[string]interface{}, device string) error {
	basicJSON, _ := json.Marshal(basicInfo)
	filmJSON, _ := json.Marshal(filmInfo)
	resultID := utils.GenerateUniqueID()

	_, err := DBPtr.DB.Exec(`
		INSERT INTO film_results (
			id, user_id, image_id, file_name, result_url, width, height, basic_info, film_info, device, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, resultID, userID, imageID, fileName, resultURL, width, height, string(basicJSON), string(filmJSON), device, time.Now().Format(time.RFC3339))

	if err != nil {
		return fmt.Errorf("failed to insert film result: %w", err)
	}

	return nil
}
