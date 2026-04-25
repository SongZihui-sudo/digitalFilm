<template>
  <main class="main-preview panel-card">
    <div class="main-preview__topbar">
      <div>
        <div class="main-preview__title">Preview</div>
        <div class="main-preview__subtitle">
          {{ currentImageName || '未选择图片' }}
        </div>
      </div>

      <div class="main-preview__tools">
        <button
          class="primary-btn"
          @click="handleSaveSettings"
          :disabled="!currentImageId || saving"
        >
          {{ saving ? '保存中...' : '保存' }}
        </button>

        <button
          class="danger-btn"
          @click="handleDeleteImage"
          :disabled="!currentImageId || saving"
        >
          {{ saving ? '删除中...' : '删除' }}
        </button>

        <button class="secondary-btn" @click="fitToScreen = !fitToScreen">
          {{ fitToScreen ? '原始大小' : '适应屏幕' }}
        </button>

        <button class="secondary-btn" @click="showCompare = !showCompare">
          {{ showCompare ? '关闭对比' : '前后对比' }}
        </button>
      </div>
    </div>

    <div class="main-preview__stage">
      <div
        v-if="currentImageUrl"
        class="preview-canvas"
        :class="{ 'is-fit': fitToScreen }"
      >
        <BeforeAfterSlider
          v-if="showCompare"
          :before-url="currentImageUrl"
          :after-url="displayImageUrl"
          :before-image-style="previewImageStyle"
          :after-image-style="previewImageStyle"
          class="preview-media"
        />

        <img
          v-else
          :src="displayImageUrl"
          :alt="currentImageName"
          class="preview-image"
          :style="previewImageStyle"
        />
      </div>

      <div v-else class="main-preview__empty">
        <div class="empty-icon">✦</div>
        <div class="empty-title">开始你的暗房工作流</div>
        <div class="empty-text">
          选择一个项目并导入照片，即可开始预览与风格化处理。
        </div>
      </div>
    </div>
  </main>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { useEditorStore } from '@/stores/editorStore'
import BeforeAfterSlider from '@/components/editor/BeforeAfterSlider.vue'
import { imageEditorService } from '@/services/ImageEditorService'
import { imageApi } from '@/api/imageApi'

const projectStore = useProjectStore()
const editorStore = useEditorStore()

const fitToScreen = ref(true)
const showCompare = ref(false)
const saving = ref(false)
const deleting = ref(false)
const loadingSettings = ref(false)

const currentImage = computed(() => projectStore.currentImage)
const currentImageId = computed(() => currentImage.value?.id || '')
const currentImageUrl = computed(() => currentImage.value?.originalUrl || '')
const currentImageName = computed(() => currentImage.value?.name || '')
const resultUrl = computed(() => editorStore.resultUrl)

// 优先显示后端生成结果，没有则显示当前原图
const displayImageUrl = computed(() => resultUrl.value || currentImageUrl.value)

// 基础参数实时预览：前端近似模拟
const previewImageStyle = computed(() => {
  const basic = editorStore.basic

  const brightness = clamp(1 + basic.exposure / 100, 0, 3)
  const contrast = clamp(1 + basic.contrast / 100, 0, 3)
  const saturate = clamp(1 + basic.saturation / 100, 0, 3)

  const warmStrength = Math.max(0, basic.temperature) / 100
  const tintRotate = basic.tint * 0.3

  const highlightBoost = basic.highlights / 200
  const shadowBoost = basic.shadows / 200

  const extraBrightness = 1 + highlightBoost * 0.08 + shadowBoost * 0.12
  const finalBrightness = clamp(brightness * extraBrightness, 0, 3)

  return {
    filter: [
      `brightness(${finalBrightness})`,
      `contrast(${contrast})`,
      `saturate(${saturate})`,
      `sepia(${warmStrength * 0.35})`,
      `hue-rotate(${tintRotate}deg)`,
    ].join(' '),
    transform: 'scale(1)',
    boxShadow: 'var(--shadow-lg)',
    borderRadius: '14px',
  }
})

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

async function handleDeleteImage() {
  if (!currentImageId.value) {
    alert('请先选择图片')
    return
  }

  deleting.value = true
  try {
    await imageEditorService.deleteImage(currentImage.value.id)
    projectStore.setCurrentImage(null)
    const images = await imageApi.listProjectImages(projectStore.getCurrentProject().id)
    projectStore.setCurrentImages(images)
    alert("当前图片已删除")

  } catch (error) {
    console.error('delete image edit settings failed:', error)
    alert('当前图片删除失败')
  } finally {
    deleting.value = false
  }
}

async function handleSaveSettings() {
  if (!currentImageId.value) {
    alert('请先选择图片')
    return
  }

  saving.value = true
  try {
    await imageEditorService.updateImageEditSettings(currentImageId.value, {
      exposure: editorStore.basic.exposure,
      contrast: editorStore.basic.contrast,
      highlights: editorStore.basic.highlights,
      shadows: editorStore.basic.shadows,
      temperature: editorStore.basic.temperature,
      tint: editorStore.basic.tint,
      saturation: editorStore.basic.saturation,
      preset: editorStore.film?.preset ?? '',
      grain: editorStore.film?.grain ?? 0,
      halation: editorStore.film?.halation ?? 0,
    })

    alert('当前的编辑已保存')
  } catch (error) {
    console.error('save image edit settings failed:', error)
    alert('当前编辑保存失败')
  } finally {
    saving.value = false
  }
}

async function loadImageSettings(imageId: string) {
  if (!imageId) return

  loadingSettings.value = true
  try {
    const settings = await imageEditorService.getImageEditSettings(imageId)

    editorStore.basic.exposure = settings.exposure ?? 0
    editorStore.basic.contrast = settings.contrast ?? 0
    editorStore.basic.highlights = settings.highlights ?? 0
    editorStore.basic.shadows = settings.shadows ?? 0
    editorStore.basic.temperature = settings.temperature ?? 0
    editorStore.basic.tint = settings.tint ?? 0
    editorStore.basic.saturation = settings.saturation ?? 0

    if (editorStore.film) {
      editorStore.film.preset = settings.preset ?? ''
      editorStore.film.grain = settings.grain ?? 0
      editorStore.film.halation = settings.halation ?? 0
    }
  } catch (error) {
    console.error('load image edit settings failed:', error)

    // 出错时回退默认值
    editorStore.basic.exposure = 0
    editorStore.basic.contrast = 0
    editorStore.basic.highlights = 0
    editorStore.basic.shadows = 0
    editorStore.basic.temperature = 0
    editorStore.basic.tint = 0
    editorStore.basic.saturation = 0

    if (editorStore.film) {
      editorStore.film.preset = ''
      editorStore.film.grain = 0
      editorStore.film.halation = 0
    }
  } finally {
    loadingSettings.value = false
  }
}

watch(
  () => currentImageId.value,
  async (newImageId) => {
    if (!newImageId) return
    await loadImageSettings(newImageId)
  },
  { immediate: true },
)
</script>

<style scoped>
.main-preview {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  padding: 18px;
}

.main-preview__topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
  gap: 16px;
}

.main-preview__title {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}

.main-preview__subtitle {
  margin-top: 6px;
  font-size: 15px;
  color: var(--text-primary);
  font-weight: 600;
  word-break: break-all;
}

.main-preview__tools {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.main-preview__stage {
  flex: 1;
  min-height: 0;
  border-radius: 18px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.025), rgba(255, 255, 255, 0.012)),
    rgba(0, 0, 0, 0.18);
  border: 1px solid rgba(255, 255, 255, 0.05);
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.preview-canvas {
  width: 100%;
  height: 100%;
  padding: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: auto;
}

.preview-canvas.is-fit {
  overflow: hidden;
}

.preview-image,
.preview-media {
  display: block;
  max-width: none;
  max-height: none;
  border-radius: 14px;
  box-shadow: var(--shadow-lg);
  transition:
    filter 120ms ease,
    transform 120ms ease,
    box-shadow 120ms ease;
}

/* 适应屏幕模式 */
.preview-canvas.is-fit .preview-image,
.preview-canvas.is-fit .preview-media {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
}

/* 原始大小模式 */
.preview-canvas:not(.is-fit) .preview-image,
.preview-canvas:not(.is-fit) .preview-media {
  width: auto;
  height: auto;
  max-width: none;
  max-height: none;
}

.main-preview__empty {
  width: 100%;
  max-width: 420px;
  text-align: center;
  color: var(--text-secondary);
  padding: 24px;
}

.empty-icon {
  width: 64px;
  height: 64px;
  margin: 0 auto 18px;
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 28px;
}

.empty-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
}

.empty-text {
  margin-top: 10px;
  line-height: 1.7;
}
</style>
