package utils

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
)

type Project struct {
	UserID    string `json:"userId"`
	ID        string `json:"id"`
	Name      string `json:"name"`
	CreatedAt string `json:"createdAt"`
	CoverURL  string `json:"coverUrl,omitempty"`
}

type ImageAsset struct {
	UserID       string `json:"userId"`
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
	UserID      string `json:"userId"`
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

type User struct {
	ID        string
	Username  string
	Email     string
	CreatedAt string
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

func GenerateUniqueID() string {
	return uuid.New().String()
}

func DeleteFileFromStaticServer(fileType, filename, StaticServer string) error {
	// 1. 确定最终要删除的文件名 (source 自动补全 .jpg，result 保持原样)
	var targetFilename string
	if fileType == "source" {
		targetFilename = fmt.Sprintf("%s.jpg", filename)
	} else if fileType == "result" {
		targetFilename = filename
	} else {
		return fmt.Errorf("invalid file type provided: %s", fileType)
	}

	// 2. 构造兼容服务端 ctx.Param 的 URL 路径
	// 清理 StaticServer 末尾可能多余的斜杠，防止出现双斜杠
	baseURL := strings.TrimRight(StaticServer, "/")

	// 拼接为 /api/images/:type/:id 的格式
	reqURL := fmt.Sprintf("%s/api/images/%s/%s", baseURL, fileType, targetFilename)

	// 3. 发送 DELETE 请求
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest(http.MethodDelete, reqURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request to static server: %w", err)
	}
	defer resp.Body.Close()

	// 4. 检查响应状态码
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to delete file from static server, status: %v", resp.StatusCode)
	}

	return nil
}
