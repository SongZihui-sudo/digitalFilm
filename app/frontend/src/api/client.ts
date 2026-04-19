import axios from 'axios';

export const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:8080',
  timeout: 30000,
});

export const static_client = axios.create({
  baseURL: 'http://127.0.0.1:6060',
  timeout: 30000,
});

export const image_client = axios.create({
  baseURL: 'http://127.0.0.1:7070',
  timeout: 30000,
});
