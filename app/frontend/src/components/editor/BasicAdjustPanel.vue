<template>
  <section class="editor-card panel-card">
    <div class="section-title">Basic Adjustments</div>
    <div class="editor-card__content">
      <BaseSlider label="曝光" v-model="editor.basic.exposure" :min="-100" :max="100" />
      <BaseSlider label="对比" v-model="editor.basic.contrast" :min="-100" :max="100" />
      <BaseSlider label="高光" v-model="editor.basic.highlights" :min="-100" :max="100" />
      <BaseSlider label="阴影" v-model="editor.basic.shadows" :min="-100" :max="100" />
      <BaseSlider label="色温" v-model="editor.basic.temperature" :min="-100" :max="100" />
      <BaseSlider label="色调" v-model="editor.basic.tint" :min="-100" :max="100" />
      <BaseSlider label="饱和度" v-model="editor.basic.saturation" :min="-100" :max="100" />
    </div>
  </section>
</template>

<script setup lang="ts">
import BaseSlider from '@/components/common/BaseSlider.vue'
import { useEditorStore } from '@/stores/editorStore'

const editor = useEditorStore()
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
