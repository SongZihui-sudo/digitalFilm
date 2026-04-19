<template>
  <aside class="left-sidebar panel-card">
    <div class="left-sidebar__top">
      <div class="section-title">Library</div>

      <button class="secondary-btn left-sidebar__upload" @click="openFileDialog">
        上传
      </button>

      <input
        ref="fileInputRef"
        type="file"
        accept="image/*"
        class="hidden-file-input"
        @change="handleFileChange"
      />
    </div>

    <div class="left-sidebar__section left-sidebar__section--card">
      <ProjectList />
    </div>

    <div class="left-sidebar__section left-sidebar__section--card left-sidebar__section--grow">
      <ImageThumbnailList />
    </div>
  </aside>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import ProjectList from '@/components/project/ProjectList.vue'
import ImageThumbnailList from '@/components/project/ImageThumbnailList.vue'
import { useProjectManager } from '@/composables/useProjectManager'

const fileInputRef = ref<HTMLInputElement | null>(null)
const { uploadImage } = useProjectManager()

function openFileDialog() {
  fileInputRef.value?.click()
}

async function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]

  if (!file) return

  await uploadImage(file)

  // 允许重复上传同一个文件
  input.value = ''
}
</script>

<style scoped>
.left-sidebar {
  min-height: 0;
  display: flex;
  flex-direction: column;
  padding: 16px;
  overflow: hidden;
}

.left-sidebar__top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.left-sidebar__upload {
  width: 86px;
}

.left-sidebar__section {
  margin-bottom: 12px;
  min-height: 0;
}

.left-sidebar__section--grow {
  flex: 1;
  overflow: hidden;
}

.left-sidebar__section--card {
  padding: 12px;
  border-radius: 16px;
  background: var(--surface-1);
  border: 1px solid var(--border-color);
  transition:
    background var(--transition-fast),
    border-color var(--transition-fast);
}

.left-sidebar__section--card:hover {
  background: var(--surface-2);
  border-color: var(--border-strong);
}

.hidden-file-input {
  display: none;
}
</style>
