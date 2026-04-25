package db

import (
	"backend/utils"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

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
	// 项目表
	projectTableSQL := `
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        cover_url TEXT DEFAULT ''
    );
    `

	// 图片表
	imageTableSQL := `
    CREATE TABLE IF NOT EXISTS image_assets (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        name TEXT NOT NULL,
        original_url TEXT NOT NULL,
        thumbnail_url TEXT DEFAULT '',
        width INTEGER DEFAULT 0,
        height INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(project_id) REFERENCES projects(id)
    );
    `

	// 图片编辑设置表
	imageSettingsTableSQL := `
    CREATE TABLE IF NOT EXISTS image_edit_settings (
        image_id TEXT PRIMARY KEY,
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
        FOREIGN KEY(image_id) REFERENCES image_assets(id)
    );
    `

	// 胶片风格结果表
	filmResultTableSQL := `
    CREATE TABLE IF NOT EXISTS film_results (
        id TEXT PRIMARY KEY,
        image_id TEXT NOT NULL,
        file_name TEXT NOT NULL,
        result_url TEXT NOT NULL,
        width INTEGER DEFAULT 0,
        height INTEGER DEFAULT 0,
        basic_info TEXT DEFAULT '',
        film_info TEXT DEFAULT '',
        device TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES image_assets(id)
    );
    `

	// 创建各表
	if _, err := DBPtr.DB.Exec(projectTableSQL); err != nil {
		DBPtr.Status = ERROR
		return fmt.Errorf("create projects table failed: %w", err)
	}

	if _, err := DBPtr.DB.Exec(imageTableSQL); err != nil {
		DBPtr.Status = ERROR
		return fmt.Errorf("create image_assets table failed: %w", err)
	}

	if _, err := DBPtr.DB.Exec(imageSettingsTableSQL); err != nil {
		DBPtr.Status = ERROR
		return fmt.Errorf("create image_edit_settings table failed: %w", err)
	}

	if _, err := DBPtr.DB.Exec(filmResultTableSQL); err != nil {
		DBPtr.Status = ERROR
		return fmt.Errorf("create film_results table failed: %w", err)
	}

	return nil
}

func (DBPtr *AppDb) LoadProjects() ([]utils.Project, error) {
	rows, err := DBPtr.DB.Query(`
		SELECT id, name, created_at, cover_url
		FROM projects
		ORDER BY created_at DESC
	`)
	if err != nil {
		return nil, fmt.Errorf("query projects failed: %w", err)
	}
	defer rows.Close()

	var projects []utils.Project
	for rows.Next() {
		var p utils.Project
		if err := rows.Scan(&p.ID, &p.Name, &p.CreatedAt, &p.CoverURL); err != nil {
			return nil, fmt.Errorf("scan project failed: %w", err)
		}
		projects = append(projects, p)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate projects failed: %w", err)
	}

	return projects, nil
}

func (DBPtr *AppDb) LoadImages() (map[string][]utils.ImageAsset, error) {
	rows, err := DBPtr.DB.Query(`
		SELECT id, project_id, name, original_url, thumbnail_url, width, height, created_at
		FROM image_assets
		ORDER BY created_at DESC
	`)
	if err != nil {
		return nil, fmt.Errorf("query image_assets failed: %w", err)
	}
	defer rows.Close()

	images := make(map[string][]utils.ImageAsset)

	for rows.Next() {
		var img utils.ImageAsset
		if err := rows.Scan(
			&img.ID,
			&img.ProjectID,
			&img.Name,
			&img.OriginalURL,
			&img.ThumbnailURL,
			&img.Width,
			&img.Height,
			&img.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan image_asset failed: %w", err)
		}

		images[img.ProjectID] = append(images[img.ProjectID], img)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate image_assets failed: %w", err)
	}

	return images, nil
}

func (DBPtr *AppDb) AppendProject(project utils.Project) error {
	_, err := DBPtr.DB.Exec(`
		INSERT INTO projects (id, name, created_at, cover_url)
		VALUES (?, ?, ?, ?)
	`,
		project.ID,
		project.Name,
		project.CreatedAt,
		project.CoverURL,
	)
	if err != nil {
		return fmt.Errorf("append project failed: %w", err)
	}

	return nil
}

func (DBPtr *AppDb) AppendImage(image utils.ImageAsset) error {
	_, err := DBPtr.DB.Exec(`
		INSERT INTO image_assets (
			id,
			project_id,
			name,
			original_url,
			thumbnail_url,
			width,
			height,
			created_at
		)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`,
		image.ID,
		image.ProjectID,
		image.Name,
		image.OriginalURL,
		image.ThumbnailURL,
		image.Width,
		image.Height,
		image.CreatedAt,
	)
	if err != nil {
		return fmt.Errorf("append image failed: %w", err)
	}

	return nil
}

func (DBPtr *AppDb) GetProjectImages(projectID string) ([]utils.ImageAsset, error) {
	rows, err := DBPtr.DB.Query(`
		SELECT id, project_id, name, original_url, thumbnail_url, width, height, created_at
		FROM image_assets
		WHERE project_id = ?
		ORDER BY created_at DESC
	`, projectID)
	if err != nil {
		return nil, fmt.Errorf("query project images failed: %w", err)
	}
	defer rows.Close()

	var images []utils.ImageAsset
	for rows.Next() {
		var img utils.ImageAsset
		if err := rows.Scan(
			&img.ID,
			&img.ProjectID,
			&img.Name,
			&img.OriginalURL,
			&img.ThumbnailURL,
			&img.Width,
			&img.Height,
			&img.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan image_asset failed: %w", err)
		}
		images = append(images, img)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate project images failed: %w", err)
	}

	return images, nil
}

func (DBPtr *AppDb) GetImageInfo(id string) (utils.ImageAsset, error) {
	var img utils.ImageAsset

	err := DBPtr.DB.QueryRow(`
		SELECT id, project_id, name, original_url, thumbnail_url, width, height, created_at
		FROM image_assets
		WHERE id = ?
	`, id).Scan(
		&img.ID,
		&img.ProjectID,
		&img.Name,
		&img.OriginalURL,
		&img.ThumbnailURL,
		&img.Width,
		&img.Height,
		&img.CreatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return utils.ImageAsset{}, fmt.Errorf("image not found: %s", id)
		}
		return utils.ImageAsset{}, fmt.Errorf("get image info failed: %w", err)
	}

	return img, nil
}

func (DBPtr *AppDb) GetImageEditSettings(imageID string) (utils.ImageEditSettings, error) {
	var settings utils.ImageEditSettings

	err := DBPtr.DB.QueryRow(`
		SELECT 
			image_id,
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
			return utils.ImageEditSettings{}, fmt.Errorf("image edit settings not found: %s", imageID)
		}
		return utils.ImageEditSettings{}, fmt.Errorf("get image edit settings failed: %w", err)
	}

	return settings, nil
}

func (DBPtr *AppDb) SaveImageEditSettings(settings utils.ImageEditSettings) error {
	_, err := DBPtr.DB.Exec(`
		INSERT INTO image_edit_settings (
			image_id,
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
		)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
		settings.ImageID,
		settings.Exposure,
		settings.Contrast,
		settings.Highlights,
		settings.Shadows,
		settings.Temperature,
		settings.Tint,
		settings.Saturation,
		settings.Preset,
		settings.Grain,
		settings.Halation,
	)
	if err != nil {
		return fmt.Errorf("save image edit settings failed: %w", err)
	}

	return nil
}

// DeleteImage 从数据库中删除图片及其关联设置
func (DBPtr *AppDb) DeleteImage(id string, StaticServer string) error {
	// 1. 查找胶片风格结果的文件名
	var filmFileName string

	// 获取胶片风格的文件名
	err := DBPtr.DB.QueryRow("SELECT file_name FROM film_results WHERE image_id = ?", id).Scan(&filmFileName)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("failed to fetch film result file name: %w", err)
	}

	// 2. 删除胶片风格结果文件
	if err == nil && filmFileName != "" {
		// Send DELETE request to static server for film result file
		if err := utils.DeleteFileFromStaticServer("result", filmFileName, StaticServer); err != nil {
			return fmt.Errorf("failed to delete film result file from static server: %w", err)
		}
	}

	// 3. 删除图片文件
	// Send DELETE request to static server for image file
	if err := utils.DeleteFileFromStaticServer("source", id, StaticServer); err != nil {
		return fmt.Errorf("failed to delete image file from static server: %w", err)
	}

	// 4. 删除胶片风格结果的数据库记录
	_, err = DBPtr.DB.Exec("DELETE FROM film_results WHERE image_id = ?", id)
	if err != nil {
		return fmt.Errorf("failed to delete film result from database: %w", err)
	}

	// 5. 删除图片的数据库记录及其相关的编辑设置
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

	// 5.1 删除图片编辑设置
	if _, err = tx.Exec("DELETE FROM image_edit_settings WHERE image_id = ?", id); err != nil {
		return fmt.Errorf("delete image settings failed: %w", err)
	}

	// 5.2 删除图片记录
	if _, err = tx.Exec("DELETE FROM image_assets WHERE id = ?", id); err != nil {
		return fmt.Errorf("delete image asset failed: %w", err)
	}

	return nil
}

func (DBPtr *AppDb) DeleteProject(id string, StaticServer string) error {
	// 1. 查询该项目下的所有图片 ID
	rows, err := DBPtr.DB.Query("SELECT id FROM image_assets WHERE project_id = ?", id)
	if err != nil {
		return fmt.Errorf("failed to query images for project %s: %w", id, err)
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

	// 2. 删除项目相关的所有图片及胶片风格结果
	for _, imgID := range imageIDs {
		if err := DBPtr.DeleteImage(imgID, StaticServer); err != nil {
			return fmt.Errorf("failed to delete image %s: %w", imgID, err)
		}
	}

	// 3. 删除与项目相关的胶片风格结果数据
	deleteFilmResultsSQL := `DELETE FROM film_results WHERE image_id IN (SELECT id FROM image_assets WHERE project_id = ?)`
	if _, err := DBPtr.DB.Exec(deleteFilmResultsSQL, id); err != nil {
		return fmt.Errorf("failed to delete film results for project %s: %w", id, err)
	}

	// 4. 删除该项目下的所有图片编辑设置
	deleteSettingsSQL := `DELETE FROM image_edit_settings WHERE image_id IN (SELECT id FROM image_assets WHERE project_id = ?)`
	if _, err := DBPtr.DB.Exec(deleteSettingsSQL, id); err != nil {
		return fmt.Errorf("failed to delete image edit settings for project %s: %w", id, err)
	}

	// 5. 删除该项目下的所有图片记录
	deleteImagesSQL := `DELETE FROM image_assets WHERE project_id = ?`
	if _, err := DBPtr.DB.Exec(deleteImagesSQL, id); err != nil {
		return fmt.Errorf("failed to delete image assets for project %s: %w", id, err)
	}

	// 6. 删除项目本身的记录
	deleteProjectSQL := `DELETE FROM projects WHERE id = ?`
	if _, err := DBPtr.DB.Exec(deleteProjectSQL, id); err != nil {
		return fmt.Errorf("failed to delete project %s: %w", id, err)
	}

	return nil
}

func (DBPtr *AppDb) SaveFilmResult(imageID, fileName, resultURL string, width, height int, basicInfo, filmInfo map[string]interface{}, device string) error {
	// 将基本信息和胶片信息转换为 JSON 字符串
	basicJSON, err := json.Marshal(basicInfo)
	if err != nil {
		return fmt.Errorf("failed to marshal basic info: %w", err)
	}

	filmJSON, err := json.Marshal(filmInfo)
	if err != nil {
		return fmt.Errorf("failed to marshal film info: %w", err)
	}

	// 创建一个唯一的 ID（可以根据需求调整生成方式）
	resultID := utils.GenerateUniqueID()

	// 保存胶片风格照片结果到数据库
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

	// 插入胶片风格结果
	insertSQL := `
    INSERT INTO film_results (id, image_id, file_name, result_url, width, height, basic_info, film_info, device, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `
	_, err = tx.Exec(insertSQL, resultID, imageID, fileName, resultURL, width, height, string(basicJSON), string(filmJSON), device, time.Now().Format(time.RFC3339))
	if err != nil {
		return fmt.Errorf("failed to insert film result: %w", err)
	}

	return nil
}
