import { ref, watch, onUnmounted, type Ref } from 'vue'
import type { BasicAdjustments } from '@/models/edit'

const MAX_DIM = 1200

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v
}

// 在一个图像对象上应用参数
function applyAdjustments(data: Uint8ClampedArray, w: number, h: number, b: BasicAdjustments) {
  const len = w * h * 4
  const exp = b.exposure / 100
  const con = b.contrast / 100
  const sat = b.saturation / 100
  const temp = b.temperature / 100
  const tint = b.tint / 100
  const hl = b.highlights / 200
  const sh = b.shadows / 200

  for (let i = 0; i < len; i += 4) {
    let r = data[i] / 255
    let g = data[i + 1] / 255
    let bv = data[i + 2] / 255
    const lum = 0.299 * r + 0.587 * g + 0.114 * bv

    /* ---- exposure ---- */
    r += r * exp
    g += g * exp
    bv += bv * exp

    /* ---- highlights / shadows ---- */
    if (hl !== 0 && lum > 0.5) {
      const f = (lum - 0.5) * 2 * hl
      r += r * f; g += g * f; bv += bv * f
    }
    if (sh !== 0 && lum < 0.5) {
      const f = (0.5 - lum) * 2 * sh
      r += r * f; g += g * f; bv += bv * f
    }

    /* ---- contrast ---- */
    r = (r - 0.5) * (1 + con) + 0.5
    g = (g - 0.5) * (1 + con) + 0.5
    bv = (bv - 0.5) * (1 + con) + 0.5

    /* ---- temperature (色温) ---- */
    if (temp > 0) {
      r += temp * 0.35
      bv -= temp * 0.35
    } else {
      const cool = -temp
      bv += cool * 0.35
      r -= cool * 0.35
    }

    /* ---- tint (色调: green ⇄ magenta) ---- */
    if (tint > 0) {
      g += tint * 0.3
      r -= tint * 0.06
      bv -= tint * 0.06
    } else {
      const mg = -tint
      r += mg * 0.18
      bv += mg * 0.18
      g -= mg * 0.06
    }

    /* ---- saturation (饱和度) ---- */
    if (sat !== 0) {
      const gray = 0.299 * r + 0.587 * g + 0.114 * bv
      const fac = 1 + sat
      r = gray + (r - gray) * fac
      g = gray + (g - gray) * fac
      bv = gray + (bv - gray) * fac
    }

    data[i]     = clamp(Math.round(r * 255), 0, 255)
    data[i + 1] = clamp(Math.round(g * 255), 0, 255)
    data[i + 2] = clamp(Math.round(bv * 255), 0, 255)
  }
}

export function useImagePreview(
  imageUrl: Ref<string>,
  getBasic: () => BasicAdjustments,
) {
  const previewSrc = ref('')
  const loading = ref(false)
  const error = ref('')

  let rawPixels: Uint8ClampedArray | null = null
  let pw = 0
  let ph = 0
  let gen = 0
  let killed = false
  let pendingSettings: BasicAdjustments | null = null
  let hasPending = false

  async function load(url: string): Promise<boolean> {
    previewSrc.value = ''
    rawPixels = null
    if (!url) return false

    loading.value = true
    error.value = ''

    return new Promise((resolve) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'

      img.onload = () => {
        let w = img.naturalWidth
        let h = img.naturalHeight
        const scale = Math.min(1, MAX_DIM / Math.max(w, h))
        w = Math.round(w * scale)
        h = Math.round(h * scale)

        const cvs = document.createElement('canvas')
        cvs.width = w
        cvs.height = h
        const ctx = cvs.getContext('2d')!
        ctx.drawImage(img, 0, 0, w, h)

        const d = ctx.getImageData(0, 0, w, h)
        rawPixels = new Uint8ClampedArray(d.data)
        pw = w; ph = h
        loading.value = false
        resolve(true)
      }

      img.onerror = () => {
        loading.value = false
        error.value = '图片加载失败'
        resolve(false)
      }

      img.src = url
    })
  }

  function doRender(b: BasicAdjustments) {
    const gid = ++gen
    killed = false

    requestAnimationFrame(() => {
      if (killed || gid !== gen) return

      const cvs = document.createElement('canvas')
      cvs.width = pw
      cvs.height = ph
      const ctx = cvs.getContext('2d')!

      const id = new ImageData(
        new Uint8ClampedArray(rawPixels!),
        pw,
        ph,
      )
      applyAdjustments(id.data, pw, ph, b)
      ctx.putImageData(id, 0, 0)

      const prev = previewSrc.value
      previewSrc.value = cvs.toDataURL('image/jpeg', 0.92)
      if (prev.startsWith('blob:')) URL.revokeObjectURL(prev)
    })
  }

  function render(b: BasicAdjustments) {
    if (!rawPixels) {
      pendingSettings = b
      hasPending = true
      return
    }
    hasPending = false
    pendingSettings = null
    doRender(b)
  }

  watch(imageUrl, async (url) => {
    killed = true
    const ok = await load(url)
    if (ok) render(hasPending ? pendingSettings! : getBasic())
  }, { immediate: true })

  watch(getBasic, (b) => render(b))

  onUnmounted(() => {
    killed = true
    const prev = previewSrc.value
    if (prev.startsWith('blob:')) URL.revokeObjectURL(prev)
  })

  return { previewSrc, loading, error }
}
