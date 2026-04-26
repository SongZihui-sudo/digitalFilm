export function generateAvatar(name: string, size = 64) {
  const initials = (name || 'U')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map(s => s[0].toUpperCase())
    .join('') || 'U'

  // pick color from name hash
  let hash = 0
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash)
    hash = hash & hash
  }
  const colors = ['#7c8cff', '#30c48d', '#ff8a65', '#f7b267', '#9ad3bc', '#b39ddb']
  const bg = colors[Math.abs(hash) % colors.length]
  const fg = '#ffffff'

  const fontSize = Math.round(size * 0.4)
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${size}' height='${size}' viewBox='0 0 ${size} ${size}'>
    <rect width='100%' height='100%' fill='${bg}' rx='${size * 0.18}' />
    <text x='50%' y='50%' dy='0.35em' text-anchor='middle' fill='${fg}' font-family='Segoe UI, Roboto, Helvetica, Arial, sans-serif' font-size='${fontSize}' font-weight='600'>${initials}</text>
  </svg>`

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`
}

export default generateAvatar
