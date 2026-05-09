<template>
  <section class="editor-card panel-card">
    <div class="section-title">AI Film Style</div>

    <div class="editor-card__content">
      <div class="preset-selector">
        <span class="label">预设</span>
        <button class="preset-trigger" @click="showLibrary = true">
          <span class="preset-trigger-text">{{ currentPresetLabel }}</span>
          <span class="preset-trigger-icon">▾</span>
        </button>
      </div>

      <BaseSlider
        label="颗粒"
        v-model="editor.film.grain"
        :min="0"
        :max="100"
      />
      <BaseSlider
        label="高光晕染"
        v-model="editor.film.halation"
        :min="0"
        :max="100"
      />
      <button class="primary-btn" @click="handleGenerate" :disabled="loading">
        {{ loading ? '加载预设中...' : '生成胶片效果' }}
      </button>
    </div>

    <Teleport to="body">
      <Transition name="fade">
        <div v-if="showLibrary" class="library-overlay" @click.self="showLibrary = false">
          <div class="library-modal">
            <div class="library-header">
              <h3>胶片预设库</h3>
              <button class="close-btn" @click="showLibrary = false">✕</button>
            </div>

            <div class="film-grid">
              <div 
                v-for="preset in presetOptions" 
                :key="preset.value"
                class="film-card"
                :class="{ active: editor.film.preset === preset.value }"
                @click="selectPreset(preset.value)"
              >
                <div class="film-card__cover">
                  <img :src="preset.image" :alt="preset.label">
                  <div class="active-indicator" v-if="editor.film.preset === preset.value">✓</div>
                </div>
                <div class="film-card__info">
                  <div class="film-name">{{ preset.label }}</div>
                  <div class="film-desc" :title="preset.description">{{ preset.description }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
    <LoginModal v-model:visible="showLoginModal" />
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import BaseSlider from '@/components/common/BaseSlider.vue'
import { useEditorStore } from '@/stores/editorStore'
import { useFilmGeneration } from '@/composables/useFilmGeneration'
import { useUserStore } from '@/stores/userStore'
import LoginModal from '@/components/common/LoginModal.vue'

const userStore = useUserStore()
const showLoginModal = ref(false)

const editor = useEditorStore()
const { generate, loadPresets } = useFilmGeneration()

const loading = ref(false)
const showLibrary = ref(false) 

interface PresetOption {
  label: string;
  value: string;
  image: string;
  description: string;
}

const presetOptions = ref<PresetOption[]>([])

// ==========================================
// 1. 定义本地配置映射表
// ==========================================
// 根据胶片的 value 键值，单独配置每种胶片的图片和介绍
const localPresetConfigs: Record<string, { image?: string; description?: string }> = {
  'kodak_gold_200': {
    image: 'https://images.unsplash.com/photo-1616423640778-28d1b53229bd?q=80&w=200&auto=format&fit=crop', // 可替换为 kodakImg
    description: '偏暖色调，高光带有一丝金黄，适合阳光下的街拍与人像。'
  }
}

// 兜底配置：如果新加的胶片没有在上面配置，就用这个默认值
const DEFAULT_IMAGE = 'https://images.unsplash.com/photo-1516961642265-531546e84af2?q=80&w=200&auto=format&fit=crop'
const DEFAULT_DESC = '经典复古胶片质感，为画面增加浓郁的故事氛围。'

const currentPresetLabel = computed(() => {
  const current = presetOptions.value.find(p => p.value === editor.film.preset)
  return current ? current.label : '请选择预设'
})

async function getPresets() {
  loading.value = true
  try {
    const data = await loadPresets()

    // ==========================================
    // 2. 合并接口数据与本地配置
    // ==========================================
    presetOptions.value = (data || []).map((item: any) => {
      const val = item.value ?? item.name ?? item
      // 根据 value 从映射表中取出配置
      const localConfig = localPresetConfigs[val] || {}

      return {
        label: item.label ?? item.name ?? item,
        value: val,
        // 优先级：本地手动配置 > 后端返回的图片 > 默认兜底图
        image: localConfig.image || item.image || item.icon || DEFAULT_IMAGE,
        // 优先级：本地手动配置 > 后端返回的描述 > 默认兜底描述
        description: localConfig.description || item.description || item.desc || DEFAULT_DESC,
      }
    })

    if (!editor.film.preset && presetOptions.value.length > 0) {
      editor.film.preset = presetOptions.value[0].value
    }
  } catch (error) {
    console.error('load presets failed:', error)
  } finally {
    loading.value = false
  }
}

function selectPreset(value: string) {
  editor.film.preset = value
  showLibrary.value = false
}

async function handleGenerate() {
  if (!userStore.isLoggedIn) {
    showLoginModal.value = true
    return
  }

  await generate()
}

onMounted(() => {
  getPresets()
})
</script>

<style scoped>
.editor-card {
  padding: 16px;
}

.editor-card__content {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

/* --- 预设触发按钮样式 --- */
.preset-selector {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.preset-selector .label {
  font-size: 13px;
  color: #666;
}
.preset-trigger {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  background: #f5f5f5;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 14px;
  color: #333;
}
.preset-trigger:hover {
  background: #ebebeb;
  border-color: #ccc;
}

/* --- 悬浮弹窗基础样式 --- */
.library-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.4);
  backdrop-filter: blur(4px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 9999;
}

.library-modal {
  width: 90%;
  max-width: 800px;
  max-height: 80vh;
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.library-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid #eee;
}
.library-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #111;
}
.close-btn {
  background: transparent;
  border: none;
  font-size: 20px;
  color: #999;
  cursor: pointer;
  padding: 4px;
  border-radius: 50%;
  line-height: 1;
  transition: background 0.2s;
}
.close-btn:hover {
  background: #f0f0f0;
  color: #333;
}

/* --- 卡片网格布局 --- */
.film-grid {
  padding: 24px;
  overflow-y: auto;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 20px;
}

/* --- 单个胶片卡片样式 --- */
.film-card {
  background: #fff;
  border-radius: 12px;
  border: 1px solid #eaeaea;
  overflow: hidden;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  display: flex;
  flex-direction: column;
}

.film-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.film-card.active {
  border-color: #007aff; 
  box-shadow: 0 0 0 2px rgba(0, 122, 255, 0.2);
}

.film-card__cover {
  position: relative;
  width: 100%;
  height: 120px;
  overflow: hidden;
  background: #f5f5f5;
}
.film-card__cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}
.film-card:hover .film-card__cover img {
  transform: scale(1.05);
}

.active-indicator {
  position: absolute;
  top: 8px;
  right: 8px;
  background: #007aff;
  color: #fff;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
}

.film-card__info {
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.film-name {
  font-size: 14px;
  font-weight: 600;
  color: #222;
}

.film-desc {
  font-size: 12px;
  color: #888;
  line-height: 1.4;
  /* 超过两行显示省略号 */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* --- Vue 动画过渡效果 --- */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
