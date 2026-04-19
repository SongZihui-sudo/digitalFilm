import { createRouter, createWebHistory } from 'vue-router';
import DarkroomWorkspace from '@/views/DarkroomWorkspace.vue';

const routes = [
  {
    path: '/',
    name: 'darkroom',
    component: DarkroomWorkspace,
  },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
