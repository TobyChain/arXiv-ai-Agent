new Vue({
    el: '#app',
    data: {
        availableDates: [],
        selectedDate: '',
        papers: [],
        loading: false,
        error: null,
        searchMode: false,
        searchQuery: '',
        // 手动抓取相关
        manualMode: false,
        fetchSubjectQuery: '',
        fetchDate: '',
        selectedSubject: null,
        subjectDropdownOpen: false,
        filteredSubjects: [],
        allSubjects: [],
        fetchTaskId: null,
        fetchTaskRunning: false,
        fetchTaskComplete: false,
        fetchTaskStatus: {},
        fetchPollingInterval: null
    },
    mounted() {
        this.loadDates();
        this.loadAllSubjects();
        // 点击外部关闭下拉菜单
        document.addEventListener('click', this.handleClickOutside);
    },
    beforeDestroy() {
        document.removeEventListener('click', this.handleClickOutside);
        if (this.fetchPollingInterval) {
            clearInterval(this.fetchPollingInterval);
        }
    },
    methods: {
        async loadDates() {
            try {
                let response;
                try {
                    response = await axios.get('api/dates');
                    this.availableDates = response.data;
                } catch (e) {
                    console.log('API api/dates failed, trying static file...');
                    response = await axios.get('database/index.json');
                    this.availableDates = response.data;
                }

                if (this.availableDates.length > 0) {
                    this.selectedDate = this.availableDates[0];
                    this.loadDailyReport();
                }
            } catch (err) {
                console.error('Failed to load dates:', err);
                this.error = '无法加载日期列表。请确保 database/index.json 存在。';
            }
        },
        async loadAllSubjects() {
            try {
                const response = await axios.get('api/subjects/all');
                this.allSubjects = response.data;
            } catch (err) {
                console.error('Failed to load subjects:', err);
            }
        },
        async loadDailyReport() {
            if (!this.selectedDate) return;
            
            this.loading = true;
            this.error = null;
            this.searchMode = false;
            this.manualMode = false;

            try {
                try {
                    const response = await axios.get(`api/report?date=${this.selectedDate}`);
                    this.papers = response.data;
                } catch (e) {
                    const response = await axios.get(`database/${this.selectedDate}.json`);
                    this.papers = response.data;
                }
            } catch (err) {
                console.error('Failed to load report:', err);
                this.error = `无法加载 ${this.selectedDate} 的报告数据。`;
                this.papers = [];
            } finally {
                this.loading = false;
            }
        },
        toggleSearchMode() {
            this.searchMode = !this.searchMode;
            this.manualMode = false;
            if (!this.searchMode) {
                this.loadDailyReport();
            } else {
                this.papers = [];
                this.error = null;
            }
        },
        toggleManualMode() {
            this.manualMode = !this.manualMode;
            this.searchMode = false;
            if (!this.manualMode) {
                this.loadDailyReport();
            } else {
                this.papers = [];
                this.error = null;
                this.resetFetchTask();
            }
        },
        async performSearch() {
            if (!this.searchQuery.trim()) return;

            this.loading = true;
            this.error = null;

            try {
                const response = await axios.get(`api/search`, {
                    params: { query: this.searchQuery }
                });
                this.papers = response.data;
            } catch (err) {
                console.error('Search failed:', err);
                this.error = '搜索失败，请检查后端服务是否运行。';
            } finally {
                this.loading = false;
            }
        },
        // 手动抓取功能
        searchSubjects() {
            if (this.fetchSubjectQuery.length < 2) {
                this.filteredSubjects = [];
                return;
            }
            
            const query = this.fetchSubjectQuery.toLowerCase();
            this.filteredSubjects = this.allSubjects.filter(s => 
                s.name.toLowerCase().includes(query) || s.code.toLowerCase().includes(query)
            ).slice(0, 10);
            
            this.subjectDropdownOpen = true;
        },
        selectSubject(subject) {
            this.selectedSubject = subject;
            this.fetchSubjectQuery = subject.name;
            this.subjectDropdownOpen = false;
            this.filteredSubjects = [];
        },
        clearSubject() {
            this.selectedSubject = null;
            this.fetchSubjectQuery = '';
        },
        handleClickOutside(event) {
            if (!event.target.closest('.relative')) {
                this.subjectDropdownOpen = false;
            }
        },
        async startFetchTask() {
            if (!this.selectedSubject) return;
            
            this.fetchTaskRunning = true;
            this.fetchTaskComplete = false;
            this.fetchTaskStatus = { progress: 0, message: '正在启动任务...' };
            
            try {
                const response = await axios.post('api/fetch', {
                    subject: this.selectedSubject.name,
                    date: this.fetchDate || null
                });
                
                this.fetchTaskId = response.data.task_id;
                this.pollTaskStatus();
            } catch (err) {
                console.error('Failed to start fetch task:', err);
                this.error = '启动任务失败: ' + (err.response?.data?.detail || err.message);
                this.fetchTaskRunning = false;
            }
        },
        pollTaskStatus() {
            if (this.fetchPollingInterval) {
                clearInterval(this.fetchPollingInterval);
            }
            
            this.fetchPollingInterval = setInterval(async () => {
                try {
                    const response = await axios.get(`api/task/${this.fetchTaskId}`);
                    this.fetchTaskStatus = response.data;
                    
                    if (response.data.status === 'completed') {
                        clearInterval(this.fetchPollingInterval);
                        this.fetchTaskRunning = false;
                        this.fetchTaskComplete = true;
                    } else if (response.data.status === 'error') {
                        clearInterval(this.fetchPollingInterval);
                        this.fetchTaskRunning = false;
                        this.error = '抓取失败: ' + response.data.message;
                    }
                } catch (err) {
                    console.error('Failed to poll task status:', err);
                    clearInterval(this.fetchPollingInterval);
                    this.fetchTaskRunning = false;
                    this.error = '无法获取任务状态';
                }
            }, 2000); // 每2秒轮询一次
        },
        async viewFetchResult() {
            if (!this.fetchTaskStatus.result_file) return;
            
            this.loading = true;
            try {
                const response = await axios.get(`database/${this.fetchTaskStatus.result_file}`);
                this.papers = response.data;
                this.manualMode = false; // 关闭手动模式，显示结果
            } catch (err) {
                console.error('Failed to load fetch result:', err);
                this.error = '无法加载抓取结果';
            } finally {
                this.loading = false;
            }
        },
        resetFetchTask() {
            this.fetchTaskId = null;
            this.fetchTaskRunning = false;
            this.fetchTaskComplete = false;
            this.fetchTaskStatus = {};
            if (this.fetchPollingInterval) {
                clearInterval(this.fetchPollingInterval);
            }
        }
    }
});
