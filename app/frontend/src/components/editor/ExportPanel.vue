<template>
  <section class="panel">
    <h3>导出</h3>
    <button :disabled="!editor.resultUrl" @click="downloadImage">下载结果</button>
  </section>
</template>

<script setup lang="ts">
import { useEditorStore } from '@/stores/editorStore';

const editor = useEditorStore();

function downloadImage() {
  const img = document.querySelector(
    '#app > div > div > main > div.main-preview__stage > div > img'
  ) as HTMLImageElement | null;
  if (!img) {
    console.warn('未找到目标图片元素');
    return;
  }
  const src = img.src;
  if (!src) return;

  const a = document.createElement('a');
  a.href = src;
  a.download = 'film-result.png'; // 可自定义文件名
  a.click();
}
</script>
