<template>
  <section class="editor-card panel-card">
    <div class="section-title">AI Film Style</div>

    <div class="editor-card__content">
      <BaseSelect
        label="预设"
        v-model="editor.film.preset"
        :options="presetOptions"
      />
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
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import BaseSlider from '@/components/common/BaseSlider.vue'
import BaseSelect from '@/components/common/BaseSelect.vue'
import { useEditorStore } from '@/stores/editorStore'
import { useFilmGeneration } from '@/composables/useFilmGeneration'

const editor = useEditorStore()
const { generate } = useFilmGeneration()
const {loadPresets} = useFilmGeneration()

const loading = ref(false)
const presetOptions = ref<{ label: string; value: string }[]>([
  { label: 'Kodak gold 200', value: 'kodak_gold_200' },
])

async function getPresets() {
  loading.value = true
  try {
    const data = await loadPresets()

    presetOptions.value = (data || []).map((item: any) => ({
      label: item.label ?? item.name ?? item,
      value: item.value ?? item.name ?? item,
    }))

    if (!editor.film.preset && presetOptions.value.length > 0) {
      editor.film.preset = presetOptions.value[0].value
    }
  } catch (error) {
    console.error('load presets failed:', error)
  } finally {
    loading.value = false
  }
}

async function handleGenerate() {
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
</style>
