<template>
  <aside class="right-panel">
    <BasicAdjustPanel />
    <FilmStylePanel />
    <ExportPanel />
  </aside>
</template>

<script setup lang="ts">
import BasicAdjustPanel from '@/components/editor/BasicAdjustPanel.vue'
import FilmStylePanel from '@/components/editor/FilmStylePanel.vue'
import ExportPanel from '@/components/editor/ExportPanel.vue'
</script>

<style scoped>
.right-panel {
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow: auto;
}
</style>
