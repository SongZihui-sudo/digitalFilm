<template>
  <section class="project-list">
    <div class="project-list__header">
      <h3 class="project-list__title">项目</h3>
      <button class="secondary-btn project-list__new-btn" @click="handleCreateProject">
        新建
      </button>
    </div>

    <ul class="project-list__items">
      <li
        v-for="project in store.projects"
        :key="project.id"
        class="project-item"
        :class="{ active: store.currentProject?.id === project.id }"
        @click="selectProject(project.id)"
      >
        <div class="project-item__icon">◉</div>
        <div class="project-item__meta">
          <div class="project-item__name">{{ project.name }}</div>
          <div class="project-item__sub">Photography Project</div>
        </div>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { useProjectManager } from '@/composables/useProjectManager'

const store = useProjectStore()
const { loadProjects, createProject, loadProjectImages } = useProjectManager()

onMounted(async () => {
  await loadProjects()
})

async function handleCreateProject() {
  const name = window.prompt('请输入项目名称')
  if (!name) return
  await createProject(name)
}

async function selectProject(projectId: string) {
  const project = store.projects.find((item) => item.id === projectId) || null
  store.setCurrentProject(project)
  await loadProjectImages(projectId)
}
</script>

<style scoped>
.project-list {
  min-height: 0;
}

.project-list__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.project-list__title {
  margin: 0;
  font-size: 14px;
  color: var(--text-primary);
}

.project-list__new-btn {
  width: 72px;
}

.project-list__items {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.project-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  border-radius: 14px;
  cursor: pointer;
  background: transparent;
  border: 1px solid transparent;
  transition:
    background var(--transition-fast),
    border-color var(--transition-fast),
    transform var(--transition-fast),
    box-shadow var(--transition-fast);
}

.project-item::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 14px;
  pointer-events: none;
  opacity: 0;
  transition: opacity var(--transition-fast);
  background: linear-gradient(
    135deg,
    rgba(124, 140, 255, 0.10),
    rgba(124, 140, 255, 0.04)
  );
}

.project-item:hover {
  background: var(--bg-card-hover);
  border-color: var(--border-strong);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.project-item:hover::after {
  opacity: 1;
}

.project-item.active {
  background: var(--bg-card-active);
  border-color: rgba(124, 140, 255, 0.35);
  box-shadow: 0 10px 24px rgba(124, 140, 255, 0.10);
}

.project-item.active::after {
  opacity: 1;
}

.project-item__icon {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--surface-1);
  color: var(--accent);
  flex-shrink: 0;
  border: 1px solid var(--border-color);
  transition:
    transform var(--transition-fast),
    border-color var(--transition-fast);
}

.project-item:hover .project-item__icon {
  transform: scale(1.04);
  border-color: var(--border-strong);
}

.project-item__meta {
  min-width: 0;
}

.project-item__name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.project-item__sub {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-muted);
}
</style>
