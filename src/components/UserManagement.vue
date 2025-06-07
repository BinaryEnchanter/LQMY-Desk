<!-- 
=======================================================
文件名：UserManagement.vue
描述：用户管理界面
功能：
    - 显示当前用户列表（设备名、序列号、用户类别）
    - 支持搜索过滤用户
    - 支持更改用户类别
    - 支持删除用户
作者：李昶毅
创建时间：2025-04-20
依赖：
    - Vue 3 Composition API
    - Tauri invoke API
=======================================================
--><template>
    <div>
        <h1>用户管理</h1>
        <input type="text" v-model="searchQuery" placeholder="搜索设备名或序列号..." />
        <table>
            <thead>
                <tr>
                    <th>设备名</th>
                    <th>设备序列号</th>
                    <th>用户类别</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="user in filteredUsers" :key="user.device_id">
                    <td>{{ user.device_name }}</td>
                    <td>{{ user.device_id }}</td>
                    <td>{{ formatUserType(user.user_type) }}</td>
                    <td>
                        <div v-if="editingUserId === user.device_id">
                            <select @change="selectCategory(user, $event)">
                                <option disabled selected value="">请选择新类别</option>
                                <option v-for="type in availableCategories(user.user_type)" :key="type" :value="type">
                                    {{ formatUserType(type) }}
                                </option>
                            </select>
                        </div>
                        <div v-else>
                            <button @click="startEditing(user.device_id)">更改类别</button>
                            <button @click="deleteUser(user.device_id)">删除</button>
                        </div>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</template>

<script>
import { ref, computed, onMounted } from "vue";
import { invoke } from "@tauri-apps/api/core";

export default {
    setup() {
        // === 数据定义 ===

        // 用户列表
        const users = ref([]);

        // 搜索框内容
        const searchQuery = ref("");

        // 当前正在编辑类别的用户 device_id
        const editingUserId = ref(null);

        // 用户类别对应中文标签
        const userTypeLabels = {
            trusted: "可信",
            regular: "普通",
            blacklist: "黑名单"
        };

        /**
         * 将用户类别英文映射为中文显示
         * @param {string} type - 用户类别英文标识
         * @returns {string} 中文标签
         */
        const formatUserType = (type) => {
            return userTypeLabels[type] || "未知";
        };

        /**
         * 获取除当前类别外的可选类别列表
         * @param {string} currentType - 当前用户类别
         * @returns {string[]} 可选类别数组
         */
        const availableCategories = (currentType) => {
            return Object.keys(userTypeLabels).filter((t) => t !== currentType);
        };

        /**
         * 启动编辑某个用户类别
         * @param {string} deviceId - 设备序列号
         */
        function startEditing(deviceId) {
            editingUserId.value = deviceId;
        }

        /**
         * 用户选择新类别后提交更新
         * @param {Object} user - 用户对象
         * @param {Event} event - change 事件
         */
        async function selectCategory(user, event) {
            const newType = event.target.value;
            if (!newType || newType === user.user_type) {
                return; // 未选择或未更改
            }

            try {
                await invoke("update_user_type", {
                    serial: user.device_id,
                    usertype: newType
                });
                user.user_type = newType;
                editingUserId.value = null;
                alert("用户类别更新成功");
            } catch (error) {
                console.error("更新用户类别失败:", error);
            }
        }

        /**
         * 过滤用户列表，支持按设备名/序列号搜索
         */
        const filteredUsers = computed(() => {
            const query = searchQuery.value.trim().toLowerCase();
            if (!query) return users.value;

            return users.value.filter(user => {
                const name = user.device_name?.toLowerCase() || "";
                const serial = user.device_id?.toLowerCase() || "";
                return name.includes(query) || serial.includes(query);
            });
        });

        /**
         * 获取用户列表（调用后端接口）
         */
        async function fetchUsers() {
            try {
                users.value = await invoke("get_user_info");
                console.log("成功获取用户信息:", users.value);
            } catch (error) {
                console.error("获取用户列表失败:", error);
            }
        }

        /**
         * 更新用户类别（备用函数，当前未直接使用）
         * @param {Object} user - 用户对象
         */
        async function updateUser(user) {
            try {
                await invoke("update_user_type", { serial: user.device_id, category: user.user_type });
                alert("用户类别更新成功");
            } catch (error) {
                console.error("更新用户类别失败:", error);
            }
        }

        /**
         * 删除用户
         * @param {string} serial - 用户设备序列号
         */
        async function deleteUser(serial) {
            if (confirm("确定删除该用户？")) {
                try {
                    await invoke("delete_userinfo", { serial });
                    users.value = users.value.filter(u => u.device_id !== serial);
                } catch (error) {
                    console.error("删除用户失败:", error);
                }
            }
        }

        // === 生命周期钩子 ===

        /**
         * 组件挂载时自动拉取用户列表
         */
        onMounted(fetchUsers);

        // === 返回模板绑定的数据与方法 ===
        return {
            searchQuery,
            filteredUsers,
            updateUser,
            deleteUser,
            editingUserId,
            formatUserType,
            availableCategories,
            startEditing,
            selectCategory,
        };
    }
};
</script>

<style scoped>
input {
    width: 300px;
    padding: 8px;
    margin-bottom: 10px;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
}
</style>