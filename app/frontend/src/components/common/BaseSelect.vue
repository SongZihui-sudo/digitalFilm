<template>
  <div class="base-select">
    <label class="base-select__label">{{ label }}</label>
    <select class="select-base" :value="modelValue" @change="onChange">
      <option v-for="item in options" :key="item.value" :value="item.value">
        {{ item.label }}
      </option>
    </select>
  </div>
</template>

<script setup lang="ts">
interface OptionItem {
  label: string
  value: string
}

defineProps<{
  label: string
  modelValue: string
  options: OptionItem[]
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
}>()

function onChange(e: Event) {
  emit('update:modelValue', (e.target as HTMLSelectElement).value)
}
</script>

<style scoped>
.base-select {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.base-select__label {
  font-size: 13px;
  color: var(--text-secondary);
}
</style>
