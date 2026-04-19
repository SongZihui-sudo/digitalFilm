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
  if (!editor.resultUrl) return;
  const a = document.createElement('a');
  a.href = editor.resultUrl;
  a.download = 'film-result.png';
  a.click();
}
</script>
