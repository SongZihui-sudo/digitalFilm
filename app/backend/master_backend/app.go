package main

import (
	"backend/master_backend/db"
	"backend/utils"
	"fmt"
	"sync"
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

func (app *App) CloseDB() error {
	return app.db.CloseDB()
}
