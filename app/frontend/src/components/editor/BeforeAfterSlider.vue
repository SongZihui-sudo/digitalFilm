<template>
  <div class="before-after">
    <div class="before-after__container">
      <img
        v-if="beforeUrl"
        :src="beforeUrl"
        class="before-after__img before"
        :style="beforeImageStyle"
      />

      <div class="after-wrapper" :style="{ width: divider + '%' }">
        <img
          v-if="afterUrl"
          :src="afterUrl"
          class="before-after__img after"
          :style="afterImageStyle"
        />
      </div>

      <div class="before-after__divider" :style="{ left: divider + '%' }"></div>

      <input
        class="before-after__range"
        type="range"
        min="0"
        max="100"
        v-model="divider"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, withDefaults } from 'vue'

withDefaults(
  defineProps<{
    beforeUrl: string
    afterUrl: string
    beforeImageStyle?: Record<string, string | number>
    afterImageStyle?: Record<string, string | number>
  }>(),
  {
    beforeImageStyle: () => ({}),
    afterImageStyle: () => ({}),
  },
)

const divider = ref(50)
</script>

<style scoped>
.before-after {
  width: 100%;
  height: 100%;
}

.before-after__container {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.before-after__img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
}

.before {
  position: relative;
  z-index: 1;
}

.after-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  overflow: hidden;
  z-index: 2;
}

.after {
  display: block;
}

.before-after__divider {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: rgba(255, 255, 255, 0.95);
  transform: translateX(-50%);
  z-index: 3;
  box-shadow:
    0 0 0 1px rgba(0, 0, 0, 0.12),
    0 0 12px rgba(255, 255, 255, 0.35);
  pointer-events: none;
}

.before-after__range {
  position: absolute;
  bottom: 16px;
  left: 16px;
  right: 16px;
  width: calc(100% - 32px);
  z-index: 4;
}
</style>
