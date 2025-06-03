<template>
    <div>
        <h1>LQMY 凌控 桌面端</h1>
        <div class="server-status">
            <h2>连接状态</h2>
            <p :class="statusClass">{{ statusMessage }}</p>
            <div>
                <label for="ip-input"><strong>IP 地址:</strong></label>
                <input id="ip-input" v-model="inputIp" placeholder="请输入服务器 IP 地址" />
                <button @click="confirmIp">确认</button>
            </div>
            <div class="buttons">
                <button :disabled="isRunning" @click="startServer">开启服务</button>
                <button :disabled="!isRunning" @click="stopServer">关闭服务</button>
            </div>
        </div>

        <div class="server-info">
            <h2>连接信息</h2>
            <p><strong>当前服务器IP 地址:</strong> {{ serverAddress || "未获取" }}</p>
            <p><strong>连接口令:</strong> {{ connectionPassword || "无" }}</p>
            <p><strong>本机编号:</strong> {{ currentUuid }}</p>
            <button @click="fetchServerInfo">刷新服务器信息</button>
        </div>

        <div class="connectors-info">
            <h2>连接用户</h2>
            <button class="btn disconnect" @click="disconnectALL()">
                全部断开
            </button>
            <div class="user-bars">
                <div v-for="(user, idx) in orderedUsers" :key="user.device_id" class="user-bar"
                    :class="{ controller: idx === 0 && pointer < max }">
                    <div class="info">
                        <span class="name">{{ user.device_name }}</span>
                        <span class="id">{{ user.device_id }}</span>
                        <span class="type">{{ user.user_type }}</span>
                    </div>
                    <div class="actions">
                        <button class="btn disconnect" @click="disconnectUser(user)">
                            断接
                        </button>
                        <button v-if="idx === 0 && pointer < max" class="btn revoke" @click="revokeControl(user)">
                            取消控制
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { useServerStore } from "../stores/server";

export default {
    setup() {
        const serverStore = useServerStore();

        const inputIp = ref("");
        const statusMessage = computed(() => (serverStore.isRunning ? "运行中" : "未启动"));
        const statusClass = computed(() => (serverStore.isRunning ? "running" : "stopped"));

        // --- 新增：curUsersInfo 相关计算属性
        const max = computed(() => serverStore.curUsersInfo.max);
        const pointer = computed(() => serverStore.curUsersInfo.pointer);
        const usersinfo = computed(() => serverStore.curUsersInfo.usersinfo || []);

        // 按 pointer 提前排序
        const orderedUsers = computed(() => {
            const arr = usersinfo.value;
            if (pointer.value < max.value && pointer.value < arr.length) {
                const ctrl = arr[pointer.value];
                return [ctrl, ...arr.filter((_, i) => i !== pointer.value)];
            }
            return arr;
        });

        // “断接” 操作
        function disconnectUser(user) {
            invoke('disconnect_by_uuid', { uuid: user.uuid });
            console.log("断接用户：", user.uuid);
        }

        // 断开所有用户
        function disconnectALL() {
            invoke('shutdown_caputure');
            console.log("断接所有用户");
        }
        // “取消控制” 操作
        function revokeControl(user) {
            // TODO: 调用 Tauri 后端命令，比如：
            invoke("revoke_control");
            console.log("取消控制：", user.device_id);
        }

        // --- 更新 store 的方法
        serverStore.updateServerInfo = function (addr, pw, uuid, isRunning, usersinfo) {
            serverStore.serverAddress = addr;
            serverStore.connectionPassword = pw;
            serverStore.currentUuid = uuid;
            serverStore.isRunning = isRunning;
            serverStore.curUsersInfo = usersinfo;
        };

        // --- RPC 调用
        async function fetchServerInfo() {
            try {

                const result = await invoke(
                    "get_server_info"
                );
                serverStore.updateServerInfo(result[0], result[1], result[2], result[3], result[4]);
            } catch (error) {
                console.error("获取服务器信息失败:", error);
            }
        }
        async function startServer() {
            try {
                await invoke("start_server");
                serverStore.isRunning = true;
                fetchServerInfo();
            } catch (error) {
                console.error("启动服务器失败:", error);
            }
        }
        async function stopServer() {
            try {
                await invoke("stop_server");
                serverStore.isRunning = false;
                fetchServerInfo();
            } catch (error) {
                console.error("停止服务器失败:", error);
            }
        }
        async function confirmIp() {
            try {
                if (serverStore.isRunning) throw "请先断开当前服务";
                await invoke("update_server_addr", { ipaddr: inputIp.value });
                console.log("服务器地址已更新为:", inputIp.value);
            } catch (error) {
                console.error("更新服务器地址失败:", error);
                alert("更新服务器地址失败: " + error);
            }
        }

        let timerId = null;
        onMounted(() => {
            fetchServerInfo();
            timerId = setInterval(fetchServerInfo, 200);
        });
        onUnmounted(() => {
            clearInterval(timerId);
        });

        return {
            inputIp,
            statusMessage,
            statusClass,
            startServer,
            stopServer,
            fetchServerInfo,
            confirmIp,
            serverAddress: computed(() => serverStore.serverAddress),
            connectionPassword: computed(() => serverStore.connectionPassword),
            currentUuid: computed(() => serverStore.currentUuid),
            isRunning: computed(() => serverStore.isRunning),
            // --- 导出给模板使用
            max,
            pointer,
            orderedUsers,
            disconnectUser,
            revokeControl,
            disconnectALL,
        };
    },
};
</script>

<style scoped>
.server-status,
.server-info {
    background: #f8f8f8;
    padding: 20px;
    border-radius: 10px;
    display: inline-block;
    margin-top: 20px;
}

.running {
    color: green;
    font-weight: bold;
}

.stopped {
    color: red;
    font-weight: bold;
}

.buttons button {
    margin: 10px;
    padding: 10px 20px;
    font-size: 16px;
}

/* ---- 用户列表样式 ---- */
.user-list {
    background: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}

.title {
    font-size: 1.4rem;
    margin-bottom: 1rem;
}

.user-cards {
    list-style: none;
    padding: 0;
    display: grid;
    gap: 1rem;
}

.user-card {
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #fff;
    position: relative;
}

.user-card.controller {
    border-color: #4caf50;
    background-color: #e8f5e9;
}

.badge {
    position: absolute;
    top: 8px;
    right: 12px;
    background-color: #4caf50;
    color: white;
    padding: 0.2rem 0.5rem;
    font-size: 0.75rem;
    border-radius: 12px;
}

.connectors-info {
    margin-top: 2rem;
}

/* 整个列表容器 */
.user-bars {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

/* 每个用户的信息条 */
.user-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    background-color: #fff;
    border: 1px solid #ddd;
    transition: background-color 0.2s, border-color 0.2s;
}

/* 控制中用户高亮 */
.user-bar.controller {
    border-color: #4caf50;
    background-color: #e8f5e9;
}

/* 左侧显示信息 */
.user-bar .info {
    display: flex;
    gap: 1.5rem;
    font-size: 0.9rem;
}

.user-bar .info .name {
    font-weight: 500;
}

.user-bar .info .id {
    color: #666;
}

.user-bar .info .type {
    font-style: italic;
}

/* 右侧按钮容器 */
.user-bar .actions {
    display: flex;
    gap: 0.5rem;
}

/* 通用按钮样式 */
.user-bar .btn {
    padding: 0.3rem 0.6rem;
    border: none;
    border-radius: 4px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: background-color 0.2s;
}

/* 断接按钮 */
.user-bar .btn.disconnect {
    background-color: #f44336;
    color: #fff;
}

.user-bar .btn.disconnect:hover {
    background-color: #d32f2f;
}

/* 取消控制按钮 */
.user-bar .btn.revoke {
    background-color: #ff9800;
    color: #fff;
}

.user-bar .btn.revoke:hover {
    background-color: #fb8c00;
}
</style>
