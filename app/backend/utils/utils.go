package utils

import (
	"encoding/json"
	"fmt"
	"os"
)

type Project struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	CreatedAt string `json:"createdAt"`
	CoverURL  string `json:"coverUrl,omitempty"`
}

type ImageAsset struct {
	ID           string `json:"id"`
	ProjectID    string `json:"projectId"`
	Name         string `json:"name"`
	OriginalURL  string `json:"originalUrl"`
	ThumbnailURL string `json:"thumbnailUrl,omitempty"`
	Width        int    `json:"width,omitempty"`
	Height       int    `json:"height,omitempty"`
	CreatedAt    string `json:"createdAt"`
}

type ImageEditSettings struct {
	ImageID     string `json:"imageId"`
	Exposure    int    `json:"exposure"`
	Contrast    int    `json:"contrast"`
	Highlights  int    `json:"highlights"`
	Shadows     int    `json:"shadows"`
	Temperature int    `json:"temperature"`
	Tint        int    `json:"tint"`
	Saturation  int    `json:"saturation"`
	Preset      string `json:"preset"`
	Grain       int    `json:"grain"`
	Halation    int    `json:"halation"`
}

type MasterBackendConfig struct {
	Port         string   `json:"port"`
	Url          string   `json:"url"`
	DBPATH       string   `json:"dbpath"`
	StaticServer string   `json:"staticServer"`
	Presets      []string `json:"presets"`
}

func LoadConfig(path string) (*MasterBackendConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file failed: %w", err)
	}
	var cfg MasterBackendConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config file failed: %w", err)
	}
	return &cfg, nil
}
