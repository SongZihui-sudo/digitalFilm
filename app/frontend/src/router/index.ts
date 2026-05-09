import { createRouter, createWebHistory } from 'vue-router';
import DarkroomWorkspace from '@/views/DarkroomWorkspace.vue';
import AdminLogin from '@/views/AdminLogin.vue';
import AdminDashboard from '@/views/AdminDashboard.vue';

const routes = [
  {
    path: '/',
    name: 'darkroom',
    component: DarkroomWorkspace,
  },
  {
    path: '/admin/login',
    name: 'admin-login',
    component: AdminLogin,
  },
  {
    path: '/admin/dashboard',
    name: 'admin-dashboard',
    component: AdminDashboard,
  },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
