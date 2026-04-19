<template>
  <div class="thumb-item" :class="{ active }">
    <div class="thumb-item__image">
      <img :src="image.thumbnailUrl || image.originalUrl" />
    </div>

    <div class="thumb-item__meta">
      <div class="thumb-item__name">{{ image.name }}</div>
      <div class="thumb-item__sub">Ready for editing</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ImageAsset } from '@/models/image'

defineProps<{
  image: ImageAsset
  active: boolean
}>()
</script>

<style scoped>
.thumb-item {
  position: relative;
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 10px;
  border-radius: 14px;
  background: transparent;
  cursor: pointer;
  border: 1px solid transparent;
  transition:
    background var(--transition-fast),
    border-color var(--transition-fast),
    transform var(--transition-fast),
    box-shadow var(--transition-fast);
}

.thumb-item::after {
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
    rgba(48, 196, 141, 0.05)
  );
}

.thumb-item:hover {
  background: var(--bg-card-hover);
  border-color: var(--border-strong);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.thumb-item:hover::after {
  opacity: 1;
}

.thumb-item.active {
  background: var(--bg-card-active);
  border-color: rgba(124, 140, 255, 0.35);
  box-shadow: 0 10px 24px rgba(124, 140, 255, 0.12);
}

.thumb-item.active::after {
  opacity: 1;
}

.thumb-item__image {
  width: 52px;
  height: 52px;
  border-radius: 12px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.04);
  flex-shrink: 0;
  border: 1px solid var(--border-color);
  transition:
    border-color var(--transition-fast),
    transform var(--transition-fast);
}

.thumb-item:hover .thumb-item__image {
  border-color: var(--border-strong);
  transform: scale(1.02);
}

.thumb-item__image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.thumb-item__meta {
  min-width: 0;
}

.thumb-item__name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.thumb-item__sub {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-muted);
}
</style>
