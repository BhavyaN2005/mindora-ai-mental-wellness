/**
 * Mindora - Client-Side Application Engine
 * Complete full-stack integration: Dual-Theme, Auth, Conversations, and Multi-Turn Chat.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------------------------
    // DOM Element References
    // ----------------------------------------------------------------------
    const sidebar = document.getElementById("sidebar");
    const sidebarOverlay = document.getElementById("sidebarOverlay");
    const toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
    const closeSidebarBtn = document.getElementById("closeSidebarBtn");
    const newChatBtn = document.getElementById("newChatBtn");
    const searchConversationsInput = document.getElementById("searchConversationsInput");
    const conversationsList = document.getElementById("conversationsList");
    const historyEmpty = document.getElementById("historyEmpty");
    const clearAllHistoryBtn = document.getElementById("clearAllHistoryBtn");

    const chatViewport = document.getElementById("chatViewport");
    const welcomeScreen = document.getElementById("welcomeScreen");
    const messageStream = document.getElementById("messageStream");
    const typingIndicator = document.getElementById("typingIndicator");
    const chatForm = document.getElementById("chatForm");
    const messageInput = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const clearConversationBtn = document.getElementById("clearConversationBtn");
    const promptPills = document.querySelectorAll(".prompt-pill");

    // Header & User Menu
    const userMenuBtn = document.getElementById("userMenuBtn");
    const userDropdown = document.getElementById("userDropdown");
    const userAvatarInitial = document.getElementById("userAvatarInitial");
    const dropdownUsername = document.getElementById("dropdownUsername");
    const dropdownEmail = document.getElementById("dropdownEmail");
    const authActionBtn = document.getElementById("authActionBtn");
    const authActionText = document.getElementById("authActionText");

    // Theme Switchers
    const themeLightBtn = document.getElementById("themeLightBtn");
    const themeDarkBtn = document.getElementById("themeDarkBtn");
    const themeSystemBtn = document.getElementById("themeSystemBtn");
    const themeRadioInputs = document.querySelectorAll('input[name="themeRadio"]');

    // Modals
    const emergencyModal = document.getElementById("emergencyModal");
    const openEmergencyBtn = document.getElementById("openEmergencyBtn");
    const closeEmergencyModalBtn = document.getElementById("closeEmergencyModalBtn");
    const closeEmergencyModalBottomBtn = document.getElementById("closeEmergencyModalBottomBtn");

    const authModal = document.getElementById("authModal");
    const closeAuthModalBtn = document.getElementById("closeAuthModalBtn");
    const tabLoginBtn = document.getElementById("tabLoginBtn");
    const tabRegisterBtn = document.getElementById("tabRegisterBtn");
    const loginForm = document.getElementById("loginForm");
    const registerForm = document.getElementById("registerForm");
    const authErrorBanner = document.getElementById("authErrorBanner");
    const authModalTitle = document.getElementById("authModalTitle");

    const profileModal = document.getElementById("profileModal");
    const openProfileModalBtn = document.getElementById("openProfileModalBtn");
    const closeProfileModalBtn = document.getElementById("closeProfileModalBtn");
    const closeProfileModalBottomBtn = document.getElementById("closeProfileModalBottomBtn");
    const profileStatusVal = document.getElementById("profileStatusVal");
    const profileEmailVal = document.getElementById("profileEmailVal");
    const clearAllChatsModalBtn = document.getElementById("clearAllChatsModalBtn");
    const deleteAccountBtn = document.getElementById("deleteAccountBtn");

    const aboutModal = document.getElementById("aboutModal");
    const openAboutModalBtn = document.getElementById("openAboutModalBtn");
    const closeAboutModalBtn = document.getElementById("closeAboutModalBtn");
    const closeAboutModalBottomBtn = document.getElementById("closeAboutModalBottomBtn");

    // ----------------------------------------------------------------------
    // State Management
    // ----------------------------------------------------------------------
    let currentUser = null;
    let activeConversationId = null;
    let conversations = [];
    let isBotTyping = false;

    // ----------------------------------------------------------------------
    // 1. Theme Management (Light, Dark, System)
    // ----------------------------------------------------------------------
    function initTheme() {
        const savedTheme = localStorage.getItem("mindora_theme") || "system";
        applyTheme(savedTheme, false);

        // System theme change listener
        const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
        mediaQuery.addEventListener("change", (e) => {
            const currentPref = localStorage.getItem("mindora_theme") || "system";
            if (currentPref === "system") {
                document.documentElement.setAttribute("data-resolved-theme", e.matches ? "dark" : "light");
            }
        });
    }

    function applyTheme(theme, saveToServer = true) {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("mindora_theme", theme);

        if (theme === "system") {
            const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
            document.documentElement.setAttribute("data-resolved-theme", prefersDark ? "dark" : "light");
        } else {
            document.documentElement.setAttribute("data-resolved-theme", theme);
        }

        // Update Theme Button Active States
        [themeLightBtn, themeDarkBtn, themeSystemBtn].forEach(btn => {
            if (btn) {
                btn.classList.toggle("active", btn.getAttribute("data-theme") === theme);
            }
        });

        // Update Profile Radio Buttons
        themeRadioInputs.forEach(radio => {
            radio.checked = (radio.value === theme);
        });

        if (saveToServer && currentUser) {
            fetch("/api/auth/settings", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ theme_pref: theme })
            }).catch(e => console.warn("Could not sync theme with server", e));
        }
    }

    [themeLightBtn, themeDarkBtn, themeSystemBtn].forEach(btn => {
        if (btn) {
            btn.addEventListener("click", () => {
                applyTheme(btn.getAttribute("data-theme"));
            });
        }
    });

    themeRadioInputs.forEach(radio => {
        radio.addEventListener("change", (e) => {
            applyTheme(e.target.value);
        });
    });

    // ----------------------------------------------------------------------
    // 2. Authentication & User Status
    // ----------------------------------------------------------------------
    async function checkAuthStatus() {
        try {
            const res = await fetch("/api/auth/me");
            const data = await res.json();

            if (data.authenticated && data.user) {
                currentUser = data.user;
                userAvatarInitial.textContent = currentUser.username.charAt(0).toUpperCase();
                dropdownUsername.textContent = currentUser.username;
                dropdownEmail.textContent = currentUser.email;
                profileStatusVal.textContent = `Member (${currentUser.username})`;
                profileEmailVal.textContent = currentUser.email;
                authActionText.textContent = "Sign Out";
                deleteAccountBtn.classList.remove("hidden");

                if (data.theme_pref && data.theme_pref !== localStorage.getItem("mindora_theme")) {
                    applyTheme(data.theme_pref, false);
                }
            } else {
                currentUser = null;
                userAvatarInitial.textContent = "G";
                dropdownUsername.textContent = "Guest User";
                dropdownEmail.textContent = "Conversations stored locally";
                profileStatusVal.textContent = "Guest Session";
                profileEmailVal.textContent = "Not registered";
                authActionText.textContent = "Sign In / Register";
                deleteAccountBtn.classList.add("hidden");
            }
        } catch (err) {
            console.warn("Could not verify user status", err);
        }
    }

    // Toggle Dropdown
    userMenuBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        userDropdown.classList.toggle("hidden");
    });

    document.addEventListener("click", (e) => {
        if (!userMenuBtn.contains(e.target) && !userDropdown.contains(e.target)) {
            userDropdown.classList.add("hidden");
        }
    });

    authActionBtn.addEventListener("click", () => {
        userDropdown.classList.add("hidden");
        if (currentUser) {
            // Sign Out
            if (confirm("Are you sure you want to sign out?")) {
                fetch("/api/auth/logout", { method: "POST" })
                    .then(() => {
                        window.location.reload();
                    });
            }
        } else {
            openAuthModal();
        }
    });

    function openAuthModal(mode = "login") {
        authErrorBanner.classList.add("hidden");
        authErrorBanner.textContent = "";
        authModal.classList.remove("hidden");
        switchAuthTab(mode);
    }

    function switchAuthTab(mode) {
        if (mode === "login") {
            tabLoginBtn.classList.add("active");
            tabRegisterBtn.classList.remove("active");
            loginForm.classList.remove("hidden");
            registerForm.classList.add("hidden");
            authModalTitle.textContent = "Sign In to Mindora";
        } else {
            tabRegisterBtn.classList.add("active");
            tabLoginBtn.classList.remove("active");
            registerForm.classList.remove("hidden");
            loginForm.classList.add("hidden");
            authModalTitle.textContent = "Create an Account";
        }
    }

    tabLoginBtn.addEventListener("click", () => switchAuthTab("login"));
    tabRegisterBtn.addEventListener("click", () => switchAuthTab("register"));
    closeAuthModalBtn.addEventListener("click", () => authModal.classList.add("hidden"));

    // Login Form Submit
    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        authErrorBanner.classList.add("hidden");
        const login = document.getElementById("loginIdentifier").value.trim();
        const password = document.getElementById("loginPassword").value;

        try {
            const res = await fetch("/api/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ login, password })
            });
            const data = await res.json();
            if (data.success) {
                authModal.classList.add("hidden");
                await checkAuthStatus();
                await loadConversations();
            } else {
                authErrorBanner.textContent = data.error || "Login failed.";
                authErrorBanner.classList.remove("hidden");
            }
        } catch (err) {
            authErrorBanner.textContent = "Connection error. Please try again.";
            authErrorBanner.classList.remove("hidden");
        }
    });

    // Register Form Submit
    registerForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        authErrorBanner.classList.add("hidden");
        const username = document.getElementById("regUsername").value.trim();
        const email = document.getElementById("regEmail").value.trim();
        const password = document.getElementById("regPassword").value;

        try {
            const res = await fetch("/api/auth/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, email, password })
            });
            const data = await res.json();
            if (data.success) {
                authModal.classList.add("hidden");
                await checkAuthStatus();
                await loadConversations();
            } else {
                authErrorBanner.textContent = data.error || "Registration failed.";
                authErrorBanner.classList.remove("hidden");
            }
        } catch (err) {
            authErrorBanner.textContent = "Connection error. Please try again.";
            authErrorBanner.classList.remove("hidden");
        }
    });

    // Delete Account
    deleteAccountBtn.addEventListener("click", async () => {
        if (confirm("Are you sure you want to permanently delete your account and all associated reflection data? This action cannot be undone.")) {
            try {
                const res = await fetch("/api/auth/account", { method: "DELETE" });
                const data = await res.json();
                if (data.success) {
                    alert("Your account has been deleted.");
                    window.location.reload();
                }
            } catch (err) {
                alert("Could not delete account. Please try again.");
            }
        }
    });

    // ----------------------------------------------------------------------
    // 3. Conversation History & Management
    // ----------------------------------------------------------------------
    async function loadConversations(searchQuery = "") {
        try {
            const url = searchQuery ? `/api/conversations?search=${encodeURIComponent(searchQuery)}` : "/api/conversations";
            const res = await fetch(url);
            const data = await res.json();

            if (data.success) {
                conversations = data.conversations;
                renderConversationsSidebar();
            }
        } catch (err) {
            console.warn("Could not load conversations", err);
        }
    }

    function renderConversationsSidebar() {
        conversationsList.innerHTML = "";

        if (!conversations || conversations.length === 0) {
            conversationsList.appendChild(historyEmpty);
            return;
        }

        // Chronological grouping
        const groups = {
            today: [],
            yesterday: [],
            previous7Days: [],
            older: []
        };

        const now = new Date();
        const todayStr = now.toDateString();
        const yesterday = new Date(now);
        yesterday.setDate(now.getDate() - 1);
        const yesterdayStr = yesterday.toDateString();
        const sevenDaysAgo = new Date(now);
        sevenDaysAgo.setDate(now.getDate() - 7);

        conversations.forEach(conv => {
            const updated = new Date(conv.updated_at);
            const convDateStr = updated.toDateString();

            if (convDateStr === todayStr) {
                groups.today.push(conv);
            } else if (convDateStr === yesterdayStr) {
                groups.yesterday.push(conv);
            } else if (updated >= sevenDaysAgo) {
                groups.previous7Days.push(conv);
            } else {
                groups.older.push(conv);
            }
        });

        const groupLabels = [
            { key: "today", title: "Today" },
            { key: "yesterday", title: "Yesterday" },
            { key: "previous7Days", title: "Previous 7 Days" },
            { key: "older", title: "Earlier" }
        ];

        groupLabels.forEach(({ key, title }) => {
            if (groups[key].length > 0) {
                const groupTitleEl = document.createElement("div");
                groupTitleEl.className = "history-group-title";
                groupTitleEl.textContent = title;
                conversationsList.appendChild(groupTitleEl);

                const listEl = document.createElement("div");
                listEl.className = "history-list";

                groups[key].forEach(conv => {
                    const item = document.createElement("div");
                    item.className = `history-item${conv.id === activeConversationId ? " active" : ""}`;
                    item.dataset.id = conv.id;

                    const contentWrap = document.createElement("div");
                    contentWrap.className = "history-item-content";

                    const titleSpan = document.createElement("span");
                    titleSpan.className = "history-item-title";
                    titleSpan.textContent = conv.title || "Conversation";

                    const metaSpan = document.createElement("span");
                    metaSpan.className = "history-item-meta";
                    const msgCountText = `${conv.message_count} ${conv.message_count === 1 ? 'message' : 'messages'}`;
                    metaSpan.textContent = msgCountText;

                    contentWrap.appendChild(titleSpan);
                    contentWrap.appendChild(metaSpan);

                    const actions = document.createElement("div");
                    actions.className = "history-item-actions";

                    // Rename button
                    const renameBtn = document.createElement("button");
                    renameBtn.className = "btn-item-action";
                    renameBtn.title = "Rename";
                    renameBtn.innerHTML = `
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                        </svg>
                    `;
                    renameBtn.addEventListener("click", (e) => {
                        e.stopPropagation();
                        renameConversation(conv.id, conv.title);
                    });

                    // Delete button
                    const deleteBtn = document.createElement("button");
                    deleteBtn.className = "btn-item-action delete";
                    deleteBtn.title = "Delete";
                    deleteBtn.innerHTML = `
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    `;
                    deleteBtn.addEventListener("click", (e) => {
                        e.stopPropagation();
                        deleteConversation(conv.id);
                    });

                    actions.appendChild(renameBtn);
                    actions.appendChild(deleteBtn);

                    item.appendChild(contentWrap);
                    item.appendChild(actions);

                    item.addEventListener("click", () => {
                        switchConversation(conv.id);
                        closeSidebarMobile();
                    });

                    listEl.appendChild(item);
                });

                conversationsList.appendChild(listEl);
            }
        });
    }

    async function switchConversation(convId) {
        if (!convId) {
            // Reset to new reflection view
            activeConversationId = null;
            messageStream.innerHTML = "";
            welcomeScreen.classList.remove("hidden");
            renderConversationsSidebar();
            return;
        }

        activeConversationId = convId;
        renderConversationsSidebar();

        try {
            const res = await fetch(`/api/conversations/${convId}`);
            const data = await res.json();

            if (data.success) {
                welcomeScreen.classList.add("hidden");
                messageStream.innerHTML = "";
                data.messages.forEach(msg => {
                    renderMessageBubble(msg.content, msg.role, msg.intent, msg.confidence, msg.is_crisis, msg.created_at);
                });
                scrollToBottom();
            }
        } catch (err) {
            console.error("Could not load conversation messages", err);
        }
    }

    async function renameConversation(convId, currentTitle) {
        const newTitle = prompt("Enter a new title for this reflection:", currentTitle);
        if (newTitle && newTitle.trim() && newTitle.trim() !== currentTitle) {
            try {
                const res = await fetch(`/api/conversations/${convId}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ title: newTitle.trim() })
                });
                if (res.ok) {
                    await loadConversations();
                }
            } catch (err) {
                alert("Failed to rename conversation.");
            }
        }
    }

    async function deleteConversation(convId) {
        if (confirm("Delete this reflection? This cannot be undone.")) {
            try {
                const res = await fetch(`/api/conversations/${convId}`, { method: "DELETE" });
                if (res.ok) {
                    if (activeConversationId === convId) {
                        switchConversation(null);
                    }
                    await loadConversations();
                }
            } catch (err) {
                alert("Failed to delete conversation.");
            }
        }
    }

    async function clearAllConversationsHandler() {
        if (confirm("Are you sure you want to clear your entire conversation history? This cannot be undone.")) {
            try {
                const res = await fetch("/api/conversations/clear", { method: "POST" });
                if (res.ok) {
                    switchConversation(null);
                    await loadConversations();
                    profileModal.classList.add("hidden");
                }
            } catch (err) {
                alert("Failed to clear history.");
            }
        }
    }

    clearAllHistoryBtn.addEventListener("click", clearAllConversationsHandler);
    clearAllChatsModalBtn.addEventListener("click", clearAllConversationsHandler);

    // New Chat Button
    newChatBtn.addEventListener("click", () => {
        switchConversation(null);
        closeSidebarMobile();
        messageInput.focus();
    });

    // Clear Current Chat Button
    clearConversationBtn.addEventListener("click", () => {
        if (activeConversationId) {
            deleteConversation(activeConversationId);
        } else {
            messageStream.innerHTML = "";
            welcomeScreen.classList.remove("hidden");
        }
    });

    // Search input
    let searchTimeout = null;
    searchConversationsInput.addEventListener("input", (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            loadConversations(e.target.value);
        }, 250);
    });

    // ----------------------------------------------------------------------
    // 4. Sidebar Responsive Controls
    // ----------------------------------------------------------------------
    toggleSidebarBtn.addEventListener("click", () => {
        sidebar.classList.toggle("open");
        sidebarOverlay.classList.toggle("open");
    });

    const closeSidebarMobile = () => {
        sidebar.classList.remove("open");
        sidebarOverlay.classList.remove("open");
    };

    closeSidebarBtn.addEventListener("click", closeSidebarMobile);
    sidebarOverlay.addEventListener("click", closeSidebarMobile);

    // ----------------------------------------------------------------------
    // 5. Chat & Multi-Turn Messaging
    // ----------------------------------------------------------------------
    // Auto-expand textarea
    messageInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 130) + "px";
    });

    // Enter to send, Shift+Enter for new line
    messageInput.addEventListener("keydown", function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Prompt pills
    promptPills.forEach(pill => {
        pill.addEventListener("click", () => {
            const promptText = pill.getAttribute("data-prompt");
            if (promptText && !isBotTyping) {
                messageInput.value = promptText;
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
    });

    // Handle Form Submit
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = messageInput.value.trim();
        if (!text || isBotTyping) return;

        // Reset input box
        messageInput.value = "";
        messageInput.style.height = "auto";

        welcomeScreen.classList.add("hidden");

        // Render user message bubble immediately
        renderMessageBubble(text, "user", null, null, false, new Date().toISOString());
        scrollToBottom();

        // Send to backend
        await sendMessage(text);
    });

    async function sendMessage(text) {
        setTyping(true);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    conversation_id: activeConversationId
                })
            });

            if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
            }

            const data = await res.json();

            // Store new conversation ID if created during first turn
            if (data.conversation_id && data.conversation_id !== activeConversationId) {
                activeConversationId = data.conversation_id;
            }

            // Refresh conversations sidebar to update title and order
            await loadConversations();

            setTimeout(() => {
                setTyping(false);
                renderMessageBubble(
                    data.response,
                    "assistant",
                    data.intent,
                    data.confidence,
                    data.is_crisis,
                    data.timestamp
                );
                scrollToBottom();
            }, 350);

        } catch (err) {
            console.error("Chat error:", err);
            setTimeout(() => {
                setTyping(false);
                renderMessageBubble(
                    "I apologize, but I encountered a communication error with the neural network backend. Please ensure the server is running properly.",
                    "assistant",
                    "error",
                    0.0,
                    false,
                    new Date().toISOString()
                );
                scrollToBottom();
            }, 250);
        }
    }

    function renderMessageBubble(content, role, intent, confidence, isCrisis, timestampStr) {
        const row = document.createElement("div");
        row.className = `message-row ${role}${isCrisis ? " crisis" : ""}`;

        if (role === "assistant") {
            const avatar = document.createElement("div");
            avatar.className = "assistant-avatar";
            avatar.textContent = "M";
            row.appendChild(avatar);
        }

        const bubbleWrap = document.createElement("div");
        bubbleWrap.className = "message-bubble-wrap";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";

        if (isCrisis) {
            const crisisTag = document.createElement("div");
            crisisTag.className = "crisis-pill-tag";
            crisisTag.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                Priority Crisis Intervention
            `;
            bubble.appendChild(crisisTag);
        }

        const textDiv = document.createElement("div");
        textDiv.innerHTML = formatMarkdown(content);
        bubble.appendChild(textDiv);

        const timeDiv = document.createElement("div");
        timeDiv.className = "message-timestamp";
        timeDiv.textContent = formatTime(timestampStr);

        bubbleWrap.appendChild(bubble);
        bubbleWrap.appendChild(timeDiv);

        row.appendChild(bubbleWrap);
        messageStream.appendChild(row);
    }

    function setTyping(typing) {
        isBotTyping = typing;
        sendBtn.disabled = typing;
        messageInput.disabled = typing;

        if (typing) {
            typingIndicator.classList.remove("hidden");
            chatViewport.appendChild(typingIndicator);
            scrollToBottom();
        } else {
            typingIndicator.classList.add("hidden");
            messageInput.focus();
        }
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatViewport.scrollTop = chatViewport.scrollHeight;
        });
    }

    function formatTime(isoStr) {
        try {
            const date = isoStr ? new Date(isoStr) : new Date();
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch (e) {
            return "";
        }
    }

    function formatMarkdown(str) {
        if (!str) return "";
        let escaped = str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // [links](url)
        escaped = escaped.replace(/\[(.*?)\]\((https?:\/\/.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        // **bold**
        escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Bullets
        escaped = escaped.replace(/\n•\s/g, '<br>• ');
        // Newlines
        escaped = escaped.replace(/\n/g, '<br>');

        return escaped;
    }

    // ----------------------------------------------------------------------
    // 6. Modals Controllers
    // ----------------------------------------------------------------------
    openEmergencyBtn.addEventListener("click", () => emergencyModal.classList.remove("hidden"));
    closeEmergencyModalBtn.addEventListener("click", () => emergencyModal.classList.add("hidden"));
    closeEmergencyModalBottomBtn.addEventListener("click", () => emergencyModal.classList.add("hidden"));

    openProfileModalBtn.addEventListener("click", () => {
        userDropdown.classList.add("hidden");
        profileModal.classList.remove("hidden");
    });
    closeProfileModalBtn.addEventListener("click", () => profileModal.classList.add("hidden"));
    closeProfileModalBottomBtn.addEventListener("click", () => profileModal.classList.add("hidden"));

    openAboutModalBtn.addEventListener("click", () => {
        userDropdown.classList.add("hidden");
        aboutModal.classList.remove("hidden");
    });
    closeAboutModalBtn.addEventListener("click", () => aboutModal.classList.add("hidden"));
    closeAboutModalBottomBtn.addEventListener("click", () => aboutModal.classList.add("hidden"));

    // Close on outside backdrop click
    [emergencyModal, authModal, profileModal, aboutModal].forEach(modal => {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) {
                modal.classList.add("hidden");
            }
        });
    });

    // ----------------------------------------------------------------------
    // Initialization
    // ----------------------------------------------------------------------
    initTheme();
    checkAuthStatus();
    loadConversations();
});
