<template>
  <section class="editor-card panel-card">
    <div class="section-title">
      <span>Basic Adjustments</span>
      <button class="reset-all-btn" @click="resetAll">重置全部</button>
    </div>
    <div class="editor-card__content">
      <BaseSlider label="曝光" v-model="editor.basic.exposure" :min="-100" :max="100" :defaultValue="0" />
      <BaseSlider label="对比" v-model="editor.basic.contrast" :min="-100" :max="100" :defaultValue="0" />
      <BaseSlider label="高光" v-model="editor.basic.highlights" :min="-100" :max="100" :defaultValue="0" />
      <BaseSlider label="阴影" v-model="editor.basic.shadows" :min="-100" :max="100" :defaultValue="0" />
      <BaseSlider label="色温" v-model="editor.basic.temperature" :min="-100" :max="100" :defaultValue="0" />
      <BaseSlider label="色调" v-model="editor.basic.tint" :min="-100" :max="100" :defaultValue="0" />
      <BaseSlider label="饱和度" v-model="editor.basic.saturation" :min="-100" :max="100" :defaultValue="0" />
    </div>
  </section>
</template>

<script setup lang="ts">
import BaseSlider from '@/components/common/BaseSlider.vue'
import { useEditorStore } from '@/stores/editorStore'

const editor = useEditorStore()

const resetAll = () => {
  // 建议在 Pinia Store 中定义一个 resetBasic 方法，或者在这里手动赋值
  editor.basic.exposure = 0
  editor.basic.contrast = 0
  editor.basic.highlights = 0
  editor.basic.shadows = 0
  editor.basic.temperature = 0
  editor.basic.tint = 0
  editor.basic.saturation = 0
}
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

.section-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.reset-all-btn {
  background: none;
  border: none;
  color: #666;
  font-size: 12px;
  cursor: pointer;
  transition: color 0.2s;
}

.reset-all-btn:hover {
  color: #409eff; /* 悬浮时变色 */
  text-decoration: underline;
}

</style>
