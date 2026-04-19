<template>
  <section class="image-list">
    <h3>照片</h3>
    <div class="image-list__items">
      <ImageThumbnailItem
        v-for="image in store.currentImages"
        :key="image.id"
        :image="image"
        :active="store.currentImage?.id === image.id"
        @click="store.setCurrentImage(image)"
      />
    </div>
  </section>
</template>

<script setup lang="ts">
import { useProjectStore } from '@/stores/projectStore';
import ImageThumbnailItem from './ImageThumbnailItem.vue';

const store = useProjectStore();
</script>
