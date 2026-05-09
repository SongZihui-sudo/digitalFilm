<template>
  <div class="admin-shell" v-if="isLoggedIn">
    <header class="admin-header">
      <div class="admin-header__brand">
        <div class="brand-dot"></div>
        <div>
          <div class="brand-title">DigitalFilm Admin</div>
          <div class="brand-subtitle">用户管理</div>
        </div>
      </div>

      <div class="admin-header__actions">
        <ThemeToggle />
        <button class="secondary-btn" @click="handleLogout">退出</button>
      </div>
    </header>

    <main class="admin-body">
      <div class="admin-toolbar panel-card">
        <h3 class="section-title">操作</h3>
        <button class="primary-btn" @click="showCreateModal = true">添加用户</button>
        <div class="admin-info">
          <span>总用户数: {{ users.length }}</span>
          <span>管理员: {{ adminCount }}</span>
        </div>
      </div>

      <div class="admin-table-card panel-card" v-if="!loading">
        <table class="admin-table">
          <thead>
            <tr>
              <th>用户名</th>
              <th>邮箱</th>
              <th>角色</th>
              <th>注册时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="user in users" :key="user.id">
              <td class="user-cell">
                <div class="user-avatar">{{ user.username.charAt(0).toUpperCase() }}</div>
                <span>{{ user.username }}</span>
              </td>
              <td class="email-cell">{{ user.email || '-' }}</td>
              <td>
                <span :class="['role-badge', user.isAdmin ? 'role-admin' : 'role-user']">
                  {{ user.isAdmin ? '管理员' : '用户' }}
                </span>
              </td>
              <td class="date-cell">{{ formatDate(user.createdAt) }}</td>
              <td>
                <div class="action-btns">
                  <button class="secondary-btn action-btn" @click="openChangePassword(user)">
                    改密
                  </button>
                  <button
                    :class="['secondary-btn action-btn', user.isAdmin ? 'admin-revoke' : 'admin-grant']"
                    @click="handleToggleAdmin(user)"
                    :disabled="user.id === currentAdminId"
                  >
                    {{ user.isAdmin ? '取消管理员' : '设为管理员' }}
                  </button>
                  <button
                    class="danger-btn action-btn"
                    @click="confirmDelete(user)"
                    :disabled="user.id === currentAdminId"
                  >
                    删除
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <div v-if="users.length === 0" class="empty-state">
          暂无用户数据
        </div>
      </div>

      <div v-if="message" :class="['message', messageType]">{{ message }}</div>
    </main>

    <!-- 创建用户弹窗 -->
    <Teleport to="body">
      <Transition name="fade">
        <div v-if="showCreateModal" class="modal-overlay" @click.self="showCreateModal = false">
          <div class="modal-content panel-card">
            <div class="modal-header">
              <h3>添加用户</h3>
              <button class="close-btn" @click="showCreateModal = false">✕</button>
            </div>

            <div class="form-item">
              <label>用户名</label>
              <input v-model="createForm.username" type="text" placeholder="至少3个字符" class="input-base" />
            </div>
            <div class="form-item">
              <label>邮箱 (可选)</label>
              <input v-model="createForm.email" type="email" placeholder="请输入邮箱" class="input-base" />
            </div>
            <div class="form-item">
              <label>密码</label>
              <input v-model="createForm.password" type="password" placeholder="至少6位" class="input-base" />
            </div>

            <div v-if="createError" class="error-msg">{{ createError }}</div>

            <button class="primary-btn full-width" @click="handleCreateUser" :disabled="creating || !isCreateValid">
              {{ creating ? '创建中...' : '确认创建' }}
            </button>
          </div>
        </div>
      </Transition>
    </Teleport>

    <!-- 修改密码弹窗 -->
    <Teleport to="body">
      <Transition name="fade">
        <div v-if="showPasswordModal" class="modal-overlay" @click.self="showPasswordModal = false">
          <div class="modal-content panel-card">
            <div class="modal-header">
              <h3>修改密码 - {{ passwordTarget?.username }}</h3>
              <button class="close-btn" @click="showPasswordModal = false">✕</button>
            </div>

            <div class="form-item">
              <label>新密码</label>
              <input v-model="newPassword" type="password" placeholder="至少6位" class="input-base" @keyup.enter="handleChangePassword" />
            </div>

            <div v-if="passwordError" class="error-msg">{{ passwordError }}</div>

            <button class="primary-btn full-width" @click="handleChangePassword" :disabled="changing || newPassword.length < 6">
              {{ changing ? '修改中...' : '确认修改' }}
            </button>
          </div>
        </div>
      </Transition>
    </Teleport>

    <!-- 确认删除弹窗 -->
    <Teleport to="body">
      <Transition name="fade">
        <div v-if="showDeleteModal" class="modal-overlay" @click.self="showDeleteModal = false">
          <div class="modal-content panel-card">
            <div class="modal-header">
              <h3>确认删除</h3>
              <button class="close-btn" @click="showDeleteModal = false">✕</button>
            </div>
            <p class="delete-confirm-text">
              确定要删除用户 <strong>{{ deleteTarget?.username }}</strong> 吗？<br/>
              此操作会级联删除该用户的所有项目和图片数据，不可恢复。
            </p>

            <div class="modal-actions">
              <button class="secondary-btn" @click="showDeleteModal = false">取消</button>
              <button class="danger-btn" @click="handleDeleteUser" :disabled="deleting">
                {{ deleting ? '删除中...' : '确认删除' }}
              </button>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { useAdminStore } from '@/stores/adminStore';
import { adminApi, type AdminUser } from '@/api/adminApi';
import ThemeToggle from '@/components/common/ThemeToggle.vue';

const router = useRouter();
const adminStore = useAdminStore();

const users = ref<AdminUser[]>([]);
const loading = ref(true);
const message = ref('');
const messageType = ref<'success' | 'error'>('success');
const currentAdminId = computed(() => adminStore.adminInfo?.id);

const isLoggedIn = computed(() => adminStore.isLoggedIn);
const adminCount = computed(() => users.value.filter(u => u.isAdmin).length);

// 创建用户
const showCreateModal = ref(false);
const creating = ref(false);
const createError = ref('');
const createForm = ref({ username: '', email: '', password: '' });
const isCreateValid = computed(() => createForm.value.username.trim().length >= 3 && createForm.value.password.trim().length >= 6);

// 修改密码
const showPasswordModal = ref(false);
const passwordTarget = ref<AdminUser | null>(null);
const newPassword = ref('');
const changing = ref(false);
const passwordError = ref('');

// 删除用户
const showDeleteModal = ref(false);
const deleteTarget = ref<AdminUser | null>(null);
const deleting = ref(false);

function showMessage(text: string, type: 'success' | 'error') {
  message.value = text;
  messageType.value = type;
  setTimeout(() => { message.value = ''; }, 3000);
}

function formatDate(dateStr: string) {
  if (!dateStr) return '-';
  const d = new Date(dateStr);
  return d.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' });
}

async function loadUsers() {
  loading.value = true;
  try {
    const data = await adminApi.listUsers();
    users.value = data.users;
  } catch (error: any) {
    showMessage('加载用户列表失败', 'error');
  } finally {
    loading.value = false;
  }
}

async function handleCreateUser() {
  if (!isCreateValid.value) return;

  creating.value = true;
  createError.value = '';

  try {
    await adminApi.createUser(createForm.value.username, createForm.value.password, createForm.value.email);
    showCreateModal.value = false;
    createForm.value = { username: '', email: '', password: '' };
    showMessage('用户创建成功', 'success');
    await loadUsers();
  } catch (error: any) {
    createError.value = error.response?.data?.error || '创建失败';
  } finally {
    creating.value = false;
  }
}

function openChangePassword(user: AdminUser) {
  passwordTarget.value = user;
  newPassword.value = '';
  passwordError.value = '';
  showPasswordModal.value = true;
}

async function handleChangePassword() {
  if (!passwordTarget.value || newPassword.value.length < 6) return;

  changing.value = true;
  passwordError.value = '';

  try {
    await adminApi.changePassword(passwordTarget.value.id, newPassword.value);
    showPasswordModal.value = false;
    showMessage('密码修改成功', 'success');
  } catch (error: any) {
    passwordError.value = error.response?.data?.error || '修改失败';
  } finally {
    changing.value = false;
  }
}

async function handleToggleAdmin(user: AdminUser) {
  try {
    const data = await adminApi.toggleAdmin(user.id);
    const idx = users.value.findIndex(u => u.id === user.id);
    if (idx !== -1) {
      users.value[idx] = data.user;
    }
    showMessage(data.user.isAdmin ? '已设为管理员' : '已取消管理员权限', 'success');
  } catch (error: any) {
    showMessage('操作失败', 'error');
  }
}

function confirmDelete(user: AdminUser) {
  deleteTarget.value = user;
  showDeleteModal.value = true;
}

async function handleDeleteUser() {
  if (!deleteTarget.value) return;

  deleting.value = true;

  try {
    await adminApi.deleteUser(deleteTarget.value.id);
    showDeleteModal.value = false;
    showMessage('用户已删除', 'success');
    await loadUsers();
  } catch (error: any) {
    showMessage('删除失败', 'error');
  } finally {
    deleting.value = false;
  }
}

function handleLogout() {
  adminStore.logout();
  router.push('/admin/login');
}

onMounted(() => {
  if (!adminStore.isLoggedIn) {
    router.push('/admin/login');
    return;
  }
  loadUsers();
});
</script>

<style scoped>
.admin-shell {
  width: 100%;
  height: 100vh;
  display: grid;
  grid-template-rows: 72px 1fr;
  background:
    radial-gradient(circle at top left, rgba(124, 140, 255, 0.08), transparent 30%),
    radial-gradient(circle at top right, rgba(48, 196, 141, 0.05), transparent 22%),
    var(--bg-app);
  transition: background 240ms ease;
  color: var(--text-primary);
}

.admin-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 22px;
  border-bottom: 1px solid var(--border-color);
  background: color-mix(in srgb, var(--bg-app) 78%, transparent);
  backdrop-filter: blur(18px);
}

.admin-header__brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-dot {
  width: 14px;
  height: 14px;
  border-radius: 999px;
  background: linear-gradient(135deg, #7c8cff, #30c48d);
  box-shadow: 0 0 24px rgba(124, 140, 255, 0.45);
}

.brand-title {
  font-size: 16px;
  font-weight: 700;
}

.brand-subtitle {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 2px;
}

.admin-header__actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.admin-body {
  padding: 16px 22px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
}

.admin-toolbar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px 20px;
}

.admin-info {
  margin-left: auto;
  display: flex;
  gap: 16px;
  font-size: 13px;
  color: var(--text-secondary);
}

.admin-table-card {
  padding: 0;
  overflow-x: auto;
}

.admin-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.admin-table th {
  text-align: left;
  padding: 14px 16px;
  color: var(--text-muted);
  font-weight: 500;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border-bottom: 1px solid var(--border-color);
  background: var(--surface-1);
}

.admin-table td {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-color);
  vertical-align: middle;
}

.admin-table tbody tr:hover {
  background: var(--surface-1);
}

.user-cell {
  display: flex;
  align-items: center;
  gap: 10px;
  border-bottom: none !important;
  padding: 14px 16px !important;
}

.user-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--accent), #6477ff);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 700;
  flex-shrink: 0;
}

.email-cell {
  color: var(--text-secondary);
}

.date-cell {
  color: var(--text-muted);
  font-size: 13px;
}

.role-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 500;
}

.role-admin {
  background: rgba(124, 140, 255, 0.18);
  color: var(--accent);
}

.role-user {
  background: var(--surface-2);
  color: var(--text-secondary);
}

.action-btns {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.action-btn {
  height: 32px !important;
  padding: 0 12px !important;
  font-size: 12px !important;
}

.admin-grant {
  background: rgba(52, 199, 89, 0.12) !important;
  color: #34c759 !important;
  border-color: rgba(52, 199, 89, 0.24) !important;
}

.admin-revoke {
  background: rgba(255, 149, 0, 0.12) !important;
  color: #ff9500 !important;
  border-color: rgba(255, 149, 0, 0.24) !important;
}

.empty-state {
  padding: 48px;
  text-align: center;
  color: var(--text-muted);
  font-size: 14px;
}

.message {
  position: fixed;
  bottom: 24px;
  right: 24px;
  padding: 12px 20px;
  border-radius: 12px;
  font-size: 14px;
  z-index: 9999;
  animation: slideUp 0.3s ease;
}

.message.success {
  background: rgba(52, 199, 89, 0.15);
  color: #34c759;
  border: 1px solid rgba(52, 199, 89, 0.3);
}

.message.error {
  background: rgba(255, 59, 48, 0.15);
  color: #ff3b30;
  border: 1px solid rgba(255, 59, 48, 0.3);
}

@keyframes slideUp {
  from { transform: translateY(20px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

/* 弹窗样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 9999;
}

.modal-content {
  width: 100%;
  max-width: 400px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-header h3 {
  margin: 0;
  font-size: 18px;
  color: var(--text-primary);
}

.close-btn {
  background: transparent;
  border: none;
  font-size: 20px;
  color: var(--text-muted);
  cursor: pointer;
}

.close-btn:hover { color: var(--text-primary); }

.form-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-item label {
  font-size: 14px;
  color: var(--text-secondary);
  font-weight: 500;
}

.error-msg {
  color: #ff3b30;
  font-size: 13px;
  background: rgba(255, 59, 48, 0.1);
  padding: 8px 12px;
  border-radius: 8px;
}

.full-width { width: 100%; }

.delete-confirm-text {
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1.6;
}

.modal-actions {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
}

.fade-enter-active,
.fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from,
.fade-leave-to { opacity: 0; }
</style>
