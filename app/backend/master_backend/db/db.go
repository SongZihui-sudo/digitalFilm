package db

import (
	"backend/utils"
	"database/sql"
	"fmt"

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
	projectTableSQL := `
	CREATE TABLE IF NOT EXISTS projects (
		id TEXT PRIMARY KEY,
		name TEXT NOT NULL,
		created_at TEXT NOT NULL,
		cover_url TEXT DEFAULT ''
	);
	`

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
