/**
 * AI Emotion & Attention Monitor - Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const globalStatusChip = document.getElementById('globalStatusChip');
    const globalStatusText = document.getElementById('globalStatusText');
    const sessionTimer = document.getElementById('sessionTimer');
    const fpsDisplay = document.getElementById('fpsDisplay');
    const faceCountBadge = document.getElementById('faceCountBadge');
    
    // Video Stream Element & Cloud Client Webcam Fallback
    const videoStream = document.getElementById('videoStream');
    const clientWebcamVideo = document.getElementById('clientWebcamVideo');
    const cameraPrompt = document.getElementById('cameraPermissionPrompt');
    const btnAllowCamera = document.getElementById('btnAllowCamera');

    let isStreamingFrames = false;
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = 320;
    offscreenCanvas.height = 240;
    const offscreenCtx = offscreenCanvas.getContext('2d');

    async function streamCloudFrames() {
        if (!isStreamingFrames || !clientWebcamVideo || clientWebcamVideo.paused || clientWebcamVideo.ended) {
            if (isStreamingFrames) setTimeout(streamCloudFrames, 100);
            return;
        }
        try {
            offscreenCtx.drawImage(clientWebcamVideo, 0, 0, 320, 240);
            const dataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.5);
            const res = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            });
            if (res.ok) {
                const metrics = await res.json();
                renderMetricsData(metrics);
            }
        } catch (e) {
            // network retry
        }
        if (isStreamingFrames) {
            setTimeout(streamCloudFrames, 80);
        }
    }

    function startClientWebcam() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
                .then(stream => {
                    if (cameraPrompt) cameraPrompt.style.display = 'none';
                    if (videoStream) videoStream.style.display = 'none';
                    if (clientWebcamVideo) {
                        clientWebcamVideo.style.display = 'block';
                        clientWebcamVideo.srcObject = stream;
                        clientWebcamVideo.play().catch(() => {});
                    }
                    isStreamingFrames = true;
                    setTimeout(streamCloudFrames, 300);
                })
                .catch(err => {
                    console.warn('Browser webcam access denied or requires user interaction:', err);
                    if (cameraPrompt) cameraPrompt.style.display = 'flex';
                });
        } else {
            if (cameraPrompt) cameraPrompt.style.display = 'flex';
        }
    }

    if (btnAllowCamera) {
        btnAllowCamera.addEventListener('click', () => {
            startClientWebcam();
        });
    }

    if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        // Cloud environment (Render/AWS/etc.)
        startClientWebcam();
    } else if (videoStream) {
        videoStream.onerror = () => {
            startClientWebcam();
        };
    }

    // HUD Elements
    const hudGaze = document.getElementById('hudGaze');
    const hudEyes = document.getElementById('hudEyes');
    const hudDistraction = document.getElementById('hudDistraction');
    const distractionAlertBanner = document.getElementById('distractionAlertBanner');
    const footerEmotion = document.getElementById('footerEmotion');
    const footerBlinks = document.getElementById('footerBlinks');

    // Gauge Elements
    const attentionScoreEl = document.getElementById('attentionScore');
    const gaugeProgress = document.getElementById('gaugeProgress');
    const attentionPill = document.getElementById('attentionPill');
    const attentionLabel = document.getElementById('attentionLabel');
    const miniStatus = document.getElementById('miniStatus');
    const miniBlinks = document.getElementById('miniBlinks');
    const miniHeadPose = document.getElementById('miniHeadPose');

    // Emotion Elements
    const confidenceBadge = document.getElementById('confidenceBadge');
    const emotionsList = ['Happy', 'Sad', 'Angry', 'Surprised', 'Fear', 'Neutral'];

    // Spatial Elements
    const gazeDirectionBadge = document.getElementById('gazeDirectionBadge');
    const radarDot = document.getElementById('radarDot');
    const gazeHCoord = document.getElementById('gazeHCoord');
    const gazeVCoord = document.getElementById('gazeVCoord');
    const yawThumb = document.getElementById('yawThumb');
    const pitchThumb = document.getElementById('pitchThumb');
    const yawVal = document.getElementById('yawVal');
    const pitchVal = document.getElementById('pitchVal');

    // Session Elements
    const statFocusedPct = document.getElementById('statFocusedPct');
    const statPartialPct = document.getElementById('statPartialPct');
    const statDistractedPct = document.getElementById('statDistractedPct');
    const statTotalFrames = document.getElementById('statTotalFrames');

    // Controls
    const btnToggleMesh = document.getElementById('btnToggleMesh');
    const btnRecalibrate = document.getElementById('btnRecalibrate');
    const btnResetSession = document.getElementById('btnResetSession');
    const btnAudioToggle = document.getElementById('btnAudioToggle');
    const audioIcon = document.getElementById('audioIcon');

    // Audio Alert State (Lazy initialization)
    let audioAlertEnabled = true;
    let lastBeepTime = 0;
    let audioCtx = null;

    function getAudioContext() {
        if (!audioCtx) {
            try {
                const AudioClass = window.AudioContext || window.webkitAudioContext;
                if (AudioClass) audioCtx = new AudioClass();
            } catch (e) {
                console.warn('AudioContext not supported or blocked:', e);
            }
        }
        return audioCtx;
    }

    function playAlertBeep() {
        if (!audioAlertEnabled) return;
        const now = Date.now();
        if (now - lastBeepTime < 3000) return; // limit beep frequency to once every 3s
        lastBeepTime = now;

        try {
            const ctx = getAudioContext();
            if (!ctx) return;
            if (ctx.state === 'suspended') {
                ctx.resume();
            }
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
            osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15); // A5
            
            gain.gain.setValueAtTime(0.2, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
            
            osc.connect(gain);
            gain.connect(ctx.destination);
            
            osc.start();
            osc.stop(ctx.currentTime + 0.35);
        } catch (e) {
            console.error('Audio alert error:', e);
        }
    }

    // ---------------- Chart.js Setup ----------------
    const chartCanvas = document.getElementById('attentionChart');
    let attentionChart = null;
    const maxDataPoints = 40;
    const chartLabels = Array(maxDataPoints).fill('');
    const chartData = Array(maxDataPoints).fill(100);

    if (chartCanvas) {
        try {
            const ctx = chartCanvas.getContext('2d');
            attentionChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartLabels,
                    datasets: [{
                        label: 'Attention %',
                        data: chartData,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 0 },
                    scales: {
                        y: {
                            min: 0,
                            max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
                        },
                        x: { display: false, grid: { display: false } }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false }
                    }
                }
            });
        } catch (e) {
            console.error('Chart init error:', e);
        }
    }

    // Gauge circumference (r=80, 2*pi*80 = 502.65)
    const gaugeCircumference = 502.65;

    function updateGauge(score, status) {
        const val = Math.max(0, Math.min(100, score || 0));
        if (attentionScoreEl) attentionScoreEl.textContent = Math.round(val);
        const offset = gaugeCircumference - (val / 100) * gaugeCircumference;
        if (gaugeProgress) gaugeProgress.style.strokeDashoffset = offset;

        let strokeColor = '#10b981';
        let glowColor = 'rgba(16, 185, 129, 0.35)';
        let labelText = 'Optimal Focus';

        if (status === 'CALIBRATING') {
            strokeColor = '#06b6d4';
            glowColor = 'rgba(6, 182, 212, 0.35)';
            labelText = 'Calibrating Camera...';
        } else if (val >= 75) {
            strokeColor = '#10b981';
            glowColor = 'rgba(16, 185, 129, 0.35)';
            labelText = 'High Focus';
        } else if (val >= 45) {
            strokeColor = '#f59e0b';
            glowColor = 'rgba(245, 158, 11, 0.35)';
            labelText = 'Moderate Attention';
        } else {
            strokeColor = '#ef4444';
            glowColor = 'rgba(239, 68, 68, 0.4)';
            labelText = 'Distracted / Disengaged';
        }

        if (gaugeProgress) {
            gaugeProgress.style.stroke = strokeColor;
            gaugeProgress.style.filter = `drop-shadow(0 0 10px ${glowColor})`;
        }
        if (attentionLabel) attentionLabel.textContent = labelText;

        // Update chart line color
        if (attentionChart && attentionChart.data.datasets[0]) {
            attentionChart.data.datasets[0].borderColor = strokeColor;
        }
    }

    function formatTime(seconds) {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return [hrs, mins, secs].map(v => v.toString().padStart(2, '0')).join(':');
    }

    function renderMetricsData(data) {
        if (!data) return;
        const status = data.status || 'CALIBRATING';
        const score = typeof data.attention_score === 'number' ? data.attention_score : 100;
        const session = data.session || {};
        const headPose = data.head_pose || { pitch: 0, yaw: 0 };
        const gazeCoords = data.gaze_coordinates || { h: 0.5, v: 0.5 };

        // Gauge & Chart Update
        updateGauge(score, status);
        if (attentionChart) {
            chartData.shift();
            chartData.push(score);
            attentionChart.update();
        }

        // Status chip
        if (globalStatusText) globalStatusText.textContent = status;
        if (globalStatusChip) {
            globalStatusChip.className = 'status-chip';
            if (status === 'FOCUSED') globalStatusChip.classList.add('status-focused');
            else if (status === 'PARTIAL') globalStatusChip.classList.add('status-partial');
            else if (status === 'DISTRACTED') globalStatusChip.classList.add('status-distracted');
            else if (status === 'NO FACE') globalStatusChip.classList.add('status-noface');
        }

        // Attention Pill & Mini Status
        if (attentionPill) {
            attentionPill.textContent = status;
            attentionPill.className = 'status-pill';
            if (status === 'FOCUSED') attentionPill.classList.add('status-focused');
            else if (status === 'PARTIAL') attentionPill.classList.add('status-partial');
            else if (status === 'DISTRACTED') attentionPill.classList.add('status-distracted');
            else if (status === 'CALIBRATING') attentionPill.classList.add('status-calibrating');
        }
        if (miniStatus) miniStatus.textContent = status;

        // Timer & FPS
        if (sessionTimer) sessionTimer.textContent = formatTime(session.duration_seconds || 0);
        if (fpsDisplay) fpsDisplay.textContent = (data.fps || 0).toFixed(1);
        if (faceCountBadge) faceCountBadge.textContent = `${data.faces_detected || 0} Face${data.faces_detected !== 1 ? 's' : ''}`;

        // HUD
        if (hudGaze) hudGaze.textContent = data.gaze_direction || 'Center';
        if (hudEyes) hudEyes.textContent = data.eyes_open ? 'Open' : 'Closed';
        if (hudDistraction) hudDistraction.textContent = `${(data.distraction_duration || 0).toFixed(1)}s`;

        // Distraction Alert Banner
        if (distractionAlertBanner) {
            if ((data.distraction_duration || 0) >= 2.5 && status === 'DISTRACTED') {
                distractionAlertBanner.classList.remove('hidden');
                playAlertBeep();
            } else {
                distractionAlertBanner.classList.add('hidden');
            }
        }

        if (footerEmotion) footerEmotion.textContent = `${data.emotion || 'Neutral'} (${data.emotion_confidence || 0}%)`;
        if (footerBlinks) footerBlinks.textContent = `${session.blinks_per_min || 0} / min`;
        if (miniBlinks) miniBlinks.textContent = session.blink_count || 0;
        if (miniHeadPose) miniHeadPose.textContent = `${(headPose.pitch || 0).toFixed(2)} / ${(headPose.yaw || 0).toFixed(2)}`;

        // Emotion spectrum update
        if (confidenceBadge) confidenceBadge.textContent = `Conf: ${data.emotion_confidence || 0}%`;
        emotionsList.forEach(em => {
            const prob = (data.emotion_probabilities && data.emotion_probabilities[em]) || 0;
            const bar = document.getElementById(`bar-${em}`);
            const valEl = document.getElementById(`val-${em}`);
            if (bar) bar.style.width = `${prob}%`;
            if (valEl) valEl.textContent = `${prob.toFixed(0)}%`;
        });

        // Spatial Gaze & Head Orientation
        if (gazeDirectionBadge) gazeDirectionBadge.textContent = data.gaze_direction || 'Looking Center';
        const h = gazeCoords.h || 0.5;
        const v = gazeCoords.v || 0.5;
        if (gazeHCoord) gazeHCoord.textContent = h.toFixed(2);
        if (gazeVCoord) gazeVCoord.textContent = v.toFixed(2);

        // Gaze Radar Dot position (clamp to 10% .. 90%)
        if (radarDot) {
            const dotLeft = Math.max(10, Math.min(90, (1.0 - h) * 100));
            const dotTop = Math.max(10, Math.min(90, v * 100));
            radarDot.style.left = `${dotLeft}%`;
            radarDot.style.top = `${dotTop}%`;
        }

        // Head pose thumbs (-1 .. +1 mapped to 0% .. 100%)
        if (yawThumb && pitchThumb) {
            const yawPct = Math.max(0, Math.min(100, (((headPose.yaw || 0) + 1.0) / 2.0) * 100));
            const pitchPct = Math.max(0, Math.min(100, (((headPose.pitch || 0) + 1.0) / 2.0) * 100));
            yawThumb.style.left = `${yawPct}%`;
            pitchThumb.style.left = `${pitchPct}%`;
        }
        if (yawVal) yawVal.textContent = (headPose.yaw || 0).toFixed(2);
        if (pitchVal) pitchVal.textContent = (headPose.pitch || 0).toFixed(2);

        // Session Aggregate
        if (statFocusedPct) statFocusedPct.textContent = `${session.focused_pct || 0}%`;
        if (statPartialPct) statPartialPct.textContent = `${session.partial_pct || 0}%`;
        if (statDistractedPct) statDistractedPct.textContent = `${session.distracted_pct || 0}%`;
        if (statTotalFrames) statTotalFrames.textContent = (session.total_frames || 0).toLocaleString();

        if (btnToggleMesh) {
            if (data.mesh_visible) btnToggleMesh.classList.add('active');
            else btnToggleMesh.classList.remove('active');
        }
    }

    // ---------------- Metrics Polling ----------------
    async function fetchMetrics() {
        if (isStreamingFrames) return; // Cloud frame streamer already updates UI
        try {
            const res = await fetch('/api/metrics');
            if (!res.ok) return;
            const data = await res.json();
            renderMetricsData(data);
        } catch (err) {
            console.error('Error in fetchMetrics:', err);
        }
    }

    // Interval polling at ~15fps for ultra-smooth live metrics
    setInterval(fetchMetrics, 70);

    // ---------------- Interactive Controls ----------------
    btnToggleMesh.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/toggle_mesh', { method: 'POST' });
            const result = await res.json();
            if (result.mesh_visible) {
                btnToggleMesh.classList.add('active');
            } else {
                btnToggleMesh.classList.remove('active');
            }
        } catch (e) {
            console.error(e);
        }
    });

    btnRecalibrate.addEventListener('click', async () => {
        try {
            await fetch('/api/recalibrate', { method: 'POST' });
            globalStatusText.textContent = 'CALIBRATING';
            globalStatusChip.className = 'status-chip';
        } catch (e) {
            console.error(e);
        }
    });

    btnResetSession.addEventListener('click', async () => {
        try {
            await fetch('/api/reset_session', { method: 'POST' });
            chartData.fill(100);
            attentionChart.update();
        } catch (e) {
            console.error(e);
        }
    });

    btnAudioToggle.addEventListener('click', () => {
        audioAlertEnabled = !audioAlertEnabled;
        audioIcon.textContent = audioAlertEnabled ? '🔔' : '🔕';
        btnAudioToggle.title = audioAlertEnabled ? 'Audio Alert: Enabled' : 'Audio Alert: Muted';
        if (audioAlertEnabled && audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
    });

    // ============================================================
    // Extended Subsystems: Tab Switching & Subsystem Controllers
    // ============================================================

    // 1. Tab Switching Navigation
    const tabButtons = document.querySelectorAll('.nav-tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');

            // Trigger data loads when switching tabs
            if (targetId === 'tab-sessions') loadSessionsList();
            if (targetId === 'tab-analytics') loadHistoricalAnalytics();
            if (targetId === 'tab-reports') loadReportsDropdown();
            if (targetId === 'tab-ml') loadMLInfo();
        });
    });

    // 2. Session Hub Controls
    const inputSessionTitle = document.getElementById('inputSessionTitle');
    const btnStartNewSession = document.getElementById('btnStartNewSession');
    const btnPauseSession = document.getElementById('btnPauseSession');
    const btnResumeSession = document.getElementById('btnResumeSession');
    const btnEndSession = document.getElementById('btnEndSession');
    const sessionManagerStatusPill = document.getElementById('sessionManagerStatusPill');
    const activeSessionTitleDisplay = document.getElementById('activeSessionTitleDisplay');
    const activeSessionDurationDisplay = document.getElementById('activeSessionDurationDisplay');
    const activeSessionSamples = document.getElementById('activeSessionSamples');
    const activeSessionAvgAttn = document.getElementById('activeSessionAvgAttn');
    const activeSessionDistractions = document.getElementById('activeSessionDistractions');
    const sessionsTableBody = document.getElementById('sessionsTableBody');
    const btnRefreshSessionsList = document.getElementById('btnRefreshSessionsList');

    // Front-Page Controls
    const btnFrontStartSession = document.getElementById('btnFrontStartSession');
    const btnFrontEndSession = document.getElementById('btnFrontEndSession');

    // Analysis Modal Elements
    const sessionAnalysisModal = document.getElementById('sessionAnalysisModal');
    const btnCloseAnalysisModal = document.getElementById('btnCloseAnalysisModal');
    const btnModalCloseDone = document.getElementById('btnModalCloseDone');
    const btnModalDownloadCsv = document.getElementById('btnModalDownloadCsv');
    const btnModalPrintableHtml = document.getElementById('btnModalPrintableHtml');
    let lastCompletedSessionUuid = null;

    if (btnFrontStartSession) {
        btnFrontStartSession.addEventListener('click', async () => {
            const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const title = `Focus Session (${timeStr})`;
            btnFrontStartSession.style.display = 'none';
            if (btnFrontEndSession) {
                btnFrontEndSession.style.display = 'inline-flex';
                btnFrontEndSession.innerHTML = '<span class="recording-indicator-dot"></span> End Session';
            }
            try {
                const res = await fetch('/api/session/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title })
                });
                await res.json();
                pollSessionStatus();
            } catch (e) {
                console.error('Error starting session:', e);
            }
        });
    }

    if (btnFrontEndSession) {
        btnFrontEndSession.addEventListener('click', async () => {
            btnFrontEndSession.innerHTML = '<span>⏳ Finalizing Report...</span>';
            try {
                const res = await fetch('/api/session/stop', { method: 'POST' });
                const data = await res.json();
                pollSessionStatus();
                if (data.status === 'success' && data.session) {
                    lastCompletedSessionUuid = data.session.session_uuid;
                    openSessionAnalysisModal(data.session.session_uuid);
                } else {
                    // Fallback to latest session if available
                    const sRes = await fetch('/api/sessions?limit=1');
                    if (sRes.ok) {
                        const sList = await sRes.json();
                        if (sList.length > 0) {
                            openSessionAnalysisModal(sList[0].session_uuid);
                        }
                    }
                }
            } catch (e) {
                console.error('Error stopping session:', e);
            } finally {
                btnFrontEndSession.innerHTML = '<span class="recording-indicator-dot"></span> End Session';
            }
        });
    }

    async function openSessionAnalysisModal(sessionUuid) {
        if (!sessionUuid) return;
        try {
            const res = await fetch(`/api/analytics/${sessionUuid}`);
            if (!res.ok) return;
            const data = await res.json();

            lastCompletedSessionUuid = sessionUuid;
            const s = data.session;
            const durMin = (data.duration_seconds / 60.0).toFixed(1);

            document.getElementById('modalSessionTitle').textContent = s.title || 'Focus Session';
            document.getElementById('modalSessionDate').textContent = new Date(s.start_time * 1000).toLocaleString();
            document.getElementById('modalHeroDuration').textContent = `${durMin}m`;
            document.getElementById('modalHeroAvgAttn').textContent = `${data.average_attention}%`;
            document.getElementById('modalHeroStability').textContent = `${data.head_pose_stats.stability_index}/100`;

            const states = data.state_distribution;
            document.getElementById('modalPctFocused').textContent = `${states.Focused || 0}%`;
            document.getElementById('modalPctPartial').textContent = `${states.Partial || 0}%`;
            document.getElementById('modalPctDistracted').textContent = `${states.Distracted || 0}%`;
            document.getElementById('modalPctNoFace').textContent = `${states["No Face"] || 0}%`;

            // Expressions list
            const exprContainer = document.getElementById('modalExpressionList');
            const exprs = data.expression_distribution;
            exprContainer.innerHTML = Object.entries(exprs)
                .sort((a, b) => b[1] - a[1])
                .map(([name, pct]) => `
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#cbd5e1;">
                        <span>${name}</span>
                        <b style="color:#38bdf8;">${pct}%</b>
                    </div>
                `).join('');

            // Gaze list
            const gazeContainer = document.getElementById('modalGazeList');
            const gazes = data.gaze_distribution;
            gazeContainer.innerHTML = Object.entries(gazes)
                .sort((a, b) => b[1] - a[1])
                .map(([dir, pct]) => `
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#cbd5e1;">
                        <span>${dir}</span>
                        <b style="color:#10b981;">${pct}%</b>
                    </div>
                `).join('');

            // Insights
            document.getElementById('modalInsightStreak').textContent = `${data.longest_focus_streak_seconds}s`;
            document.getElementById('modalInsightDistractions').textContent = `${data.events_summary.distraction_events_count} (${data.events_summary.total_distracted_seconds}s)`;
            document.getElementById('modalInsightClosures').textContent = `${data.events_summary.prolonged_closures_count} times`;

            // Open modal
            sessionAnalysisModal.classList.add('open');
        } catch (e) {
            console.error('Error loading session analysis modal:', e);
        }
    }

    function closeAnalysisModal() {
        if (sessionAnalysisModal) sessionAnalysisModal.classList.remove('open');
    }

    if (btnCloseAnalysisModal) btnCloseAnalysisModal.addEventListener('click', closeAnalysisModal);
    if (btnModalCloseDone) btnModalCloseDone.addEventListener('click', closeAnalysisModal);
    if (sessionAnalysisModal) {
        sessionAnalysisModal.addEventListener('click', (e) => {
            if (e.target === sessionAnalysisModal) closeAnalysisModal();
        });
    }

    if (btnModalDownloadCsv) {
        btnModalDownloadCsv.addEventListener('click', () => {
            if (lastCompletedSessionUuid) window.open(`/api/reports/${lastCompletedSessionUuid}/csv`, '_blank');
        });
    }

    if (btnModalPrintableHtml) {
        btnModalPrintableHtml.addEventListener('click', () => {
            if (lastCompletedSessionUuid) window.open(`/api/reports/${lastCompletedSessionUuid}/html`, '_blank');
        });
    }

    if (btnStartNewSession) {
        btnStartNewSession.addEventListener('click', async () => {
            const title = inputSessionTitle.value.trim() || 'Focus Session';
            try {
                const res = await fetch('/api/session/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title })
                });
                await res.json();
                pollSessionStatus();
                loadSessionsList();
            } catch (e) {
                console.error(e);
            }
        });
    }

    if (btnPauseSession) {
        btnPauseSession.addEventListener('click', async () => {
            await fetch('/api/session/pause', { method: 'POST' });
            pollSessionStatus();
        });
    }

    if (btnResumeSession) {
        btnResumeSession.addEventListener('click', async () => {
            await fetch('/api/session/resume', { method: 'POST' });
            pollSessionStatus();
        });
    }

    if (btnEndSession) {
        btnEndSession.addEventListener('click', async () => {
            const res = await fetch('/api/session/stop', { method: 'POST' });
            const data = await res.json();
            pollSessionStatus();
            loadSessionsList();
            if (data.status === 'success' && data.session) {
                openSessionAnalysisModal(data.session.session_uuid);
            }
        });
    }

    if (btnRefreshSessionsList) {
        btnRefreshSessionsList.addEventListener('click', loadSessionsList);
    }

    async function pollSessionStatus() {
        try {
            const res = await fetch('/api/session/status');
            if (!res.ok) return;
            const data = await res.json();

            if (data.has_active_session) {
                if (btnFrontStartSession) btnFrontStartSession.style.display = 'none';
                if (btnFrontEndSession) btnFrontEndSession.style.display = 'inline-flex';

                sessionManagerStatusPill.textContent = data.status;
                sessionManagerStatusPill.className = 'status-pill ' + (data.status === 'ACTIVE' ? 'status-focused' : 'status-partial');
                activeSessionTitleDisplay.textContent = data.title;
                activeSessionDurationDisplay.textContent = formatTime(data.duration_seconds);
                activeSessionSamples.textContent = data.samples_count;
                activeSessionAvgAttn.textContent = `${data.current_avg_attention}%`;
                activeSessionDistractions.textContent = data.distraction_events;
            } else {
                if (btnFrontStartSession) btnFrontStartSession.style.display = 'inline-flex';
                if (btnFrontEndSession) btnFrontEndSession.style.display = 'none';

                sessionManagerStatusPill.textContent = 'IDLE';
                sessionManagerStatusPill.className = 'status-pill status-calibrating';
                activeSessionTitleDisplay.textContent = 'No Active Session';
                activeSessionDurationDisplay.textContent = '00:00:00';
                activeSessionSamples.textContent = '0';
                activeSessionAvgAttn.textContent = '100%';
                activeSessionDistractions.textContent = '0';
            }
        } catch (e) {
            console.error(e);
        }
    }

    setInterval(pollSessionStatus, 1000);

    async function loadSessionsList() {
        try {
            const res = await fetch('/api/sessions');
            if (!res.ok) return;
            const sessions = await res.json();

            if (!sessions.length) {
                sessionsTableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No sessions recorded yet.</td></tr>';
                return;
            }

            sessionsTableBody.innerHTML = sessions.map(s => `
                <tr>
                    <td><b>${escapeHtml(s.title || 'Focus Session')}</b><br><small style="color:var(--text-muted);">${new Date(s.start_time * 1000).toLocaleString()}</small></td>
                    <td>${formatTime(Math.round(s.duration_seconds || 0))}</td>
                    <td style="color:${(s.avg_attention || 100) >= 75 ? '#10b981' : '#f59e0b'}; font-weight:700;">${(s.avg_attention || 100).toFixed(1)}%</td>
                    <td>${escapeHtml(s.dominant_expression || 'Neutral')}</td>
                    <td>
                        <button class="btn btn-secondary" onclick="viewSessionReport('${s.session_uuid}')" style="padding:4px 8px; font-size:0.75rem;">Report</button>
                        <button class="btn btn-danger" onclick="deleteSessionRecord('${s.session_uuid}')" style="padding:4px 8px; font-size:0.75rem;">Delete</button>
                    </td>
                </tr>
            `).join('');
        } catch (e) {
            console.error(e);
        }
    }

    window.deleteSessionRecord = async function(uuid) {
        if (!confirm('Are you sure you want to delete this recorded session?')) return;
        try {
            await fetch(`/api/sessions/${uuid}/delete`, { method: 'POST' });
            loadSessionsList();
            loadReportsDropdown();
        } catch (e) {
            console.error(e);
        }
    };

    window.viewSessionReport = function(uuid) {
        // Switch to reports tab and load
        const reportsBtn = document.querySelector('[data-tab="tab-reports"]');
        if (reportsBtn) reportsBtn.click();
        const select = document.getElementById('reportSessionSelect');
        if (select) {
            select.value = uuid;
            loadSelectedReport(uuid);
        }
    };

    // 3. Advanced Analytics & Historical Neural Trends
    let trendsChart = null;
    let aggregateDoughnut = null;

    async function loadHistoricalAnalytics() {
        try {
            const res = await fetch('/api/analytics/trends');
            if (!res.ok) return;
            const data = await res.json();

            // 1. KPI Cards
            const avgScore = data.overall_avg_attention || 100.0;
            const avgEl = document.getElementById('analyticsOverallAvg');
            if (avgEl) avgEl.textContent = `${avgScore.toFixed(1)}%`;

            const kpiStatus = document.getElementById('kpiAvgStatus');
            if (kpiStatus) {
                if (avgScore >= 75) {
                    kpiStatus.textContent = 'Optimal';
                    kpiStatus.className = 'kpi-badge badge-green';
                } else if (avgScore >= 45) {
                    kpiStatus.textContent = 'Moderate';
                    kpiStatus.className = 'kpi-badge badge-amber';
                } else {
                    kpiStatus.textContent = 'Low Focus';
                    kpiStatus.className = 'kpi-badge badge-amber';
                }
            }

            const peakEl = document.getElementById('analyticsPeakScore');
            if (peakEl) peakEl.textContent = `${(data.peak_focus_score || 100.0).toFixed(1)}%`;

            const totalTimeEl = document.getElementById('analyticsTotalTime');
            if (totalTimeEl) totalTimeEl.textContent = `${data.total_duration_minutes || 0} mins`;

            const sessionsEl = document.getElementById('analyticsCompletedSessions');
            if (sessionsEl) sessionsEl.textContent = data.total_sessions || 0;

            const distractEl = document.getElementById('analyticsDistractionCount');
            if (distractEl) distractEl.textContent = data.total_distractions_logged || 0;

            const distractTimeEl = document.getElementById('analyticsDistractionTime');
            if (distractTimeEl) distractTimeEl.textContent = `${data.total_distracted_time_seconds || 0}s total`;

            // 2. Render Curved Gradient Trends Chart
            const canvasTrends = document.getElementById('historicalTrendsChart');
            if (canvasTrends) {
                const ctxTrends = canvasTrends.getContext('2d');
                const history = data.sessions_history || [];
                const labels = history.map(s => {
                    const d = new Date(s.start_time * 1000);
                    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                });
                const scores = history.map(s => s.avg_attention);

                // Create gradient
                const grad = ctxTrends.createLinearGradient(0, 0, 0, 240);
                grad.addColorStop(0, 'rgba(56, 189, 248, 0.45)');
                grad.addColorStop(1, 'rgba(56, 189, 248, 0.02)');

                if (trendsChart) trendsChart.destroy();
                trendsChart = new Chart(ctxTrends, {
                    type: 'line',
                    data: {
                        labels: labels.length ? labels : ['No Sessions Recorded'],
                        datasets: [{
                            label: 'Focus Score %',
                            data: scores.length ? scores : [100],
                            borderColor: '#38bdf8',
                            backgroundColor: grad,
                            borderWidth: 2.5,
                            fill: true,
                            tension: 0.35,
                            pointBackgroundColor: '#38bdf8',
                            pointBorderColor: '#0f172a',
                            pointBorderWidth: 2,
                            pointRadius: scores.length ? 5 : 0,
                            pointHoverRadius: 7
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { intersect: false, mode: 'index' },
                        scales: {
                            y: {
                                min: 0,
                                max: 100,
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
                            },
                            x: {
                                grid: { color: 'rgba(255, 255, 255, 0.03)' },
                                ticks: { color: '#94a3b8', font: { size: 10 } }
                            }
                        },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                borderColor: 'rgba(56, 189, 248, 0.4)',
                                borderWidth: 1,
                                padding: 10,
                                displayColors: false,
                                callbacks: {
                                    label: (ctx) => `Focus Score: ${ctx.parsed.y.toFixed(1)}%`
                                }
                            }
                        }
                    }
                });
            }

            // 3. Render Aggregate Doughnut Chart
            const canvasDoughnut = document.getElementById('aggregateDoughnutChart');
            if (canvasDoughnut) {
                const ctxDoughnut = canvasDoughnut.getContext('2d');
                const states = data.state_distribution || { Focused: 100, Partial: 0, Distracted: 0, "No Face": 0 };
                
                if (aggregateDoughnut) aggregateDoughnut.destroy();
                aggregateDoughnut = new Chart(ctxDoughnut, {
                    type: 'doughnut',
                    data: {
                        labels: ['Focused (≥75%)', 'Partial (45-74%)', 'Distracted (<45%)', 'No Face'],
                        datasets: [{
                            data: [states.Focused || 0, states.Partial || 0, states.Distracted || 0, states["No Face"] || 0],
                            backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#64748b'],
                            borderColor: '#0f172a',
                            borderWidth: 3,
                            hoverOffset: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '68%',
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 10, padding: 8 }
                            },
                            tooltip: {
                                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                borderColor: 'rgba(255, 255, 255, 0.1)',
                                borderWidth: 1,
                                callbacks: {
                                    label: (ctx) => ` ${ctx.label}: ${ctx.parsed}%`
                                }
                            }
                        }
                    }
                });
            }

            // 4. Render All-Time Emotional Reactions
            const exprContainer = document.getElementById('analyticsExpressionBars');
            if (exprContainer) {
                const exprs = data.expression_distribution || { Neutral: 100 };
                const exprColors = {
                    Happy: '#10b981', Neutral: '#38bdf8', Sad: '#64748b',
                    Angry: '#ef4444', Surprised: '#f59e0b', Fear: '#a855f7'
                };
                const exprIcons = {
                    Happy: '😄', Neutral: '😐', Sad: '😢',
                    Angry: '😡', Surprised: '😲', Fear: '😨'
                };
                exprContainer.innerHTML = Object.entries(exprs)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, pct]) => `
                        <div class="analytics-bar-item">
                            <div class="analytics-bar-info">
                                <span>${exprIcons[name] || '😐'} ${name}</span>
                                <b style="color:${exprColors[name] || '#38bdf8'};">${pct}%</b>
                            </div>
                            <div class="progress-bar-bg" style="height:5px;">
                                <div class="progress-bar-fill" style="width:${pct}%; background:${exprColors[name] || '#38bdf8'};"></div>
                            </div>
                        </div>
                    `).join('');
            }

            // 5. Render Spatial Gaze Orientation Share
            const gazeContainer = document.getElementById('analyticsGazeBars');
            if (gazeContainer) {
                const gazes = data.gaze_distribution || { Center: 100 };
                const gazeColors = { Center: '#10b981', Right: '#38bdf8', Left: '#a855f7', Up: '#f59e0b', Down: '#64748b' };
                gazeContainer.innerHTML = Object.entries(gazes)
                    .sort((a, b) => b[1] - a[1])
                    .map(([dir, pct]) => `
                        <div class="analytics-bar-item">
                            <div class="analytics-bar-info">
                                <span>🎯 ${dir}</span>
                                <b style="color:${gazeColors[dir] || '#10b981'};">${pct}%</b>
                            </div>
                            <div class="progress-bar-bg" style="height:5px;">
                                <div class="progress-bar-fill" style="width:${pct}%; background:${gazeColors[dir] || '#10b981'};"></div>
                            </div>
                        </div>
                    `).join('');
            }

            // 6. Render Mini Recent Sessions Table
            const miniTableBody = document.getElementById('analyticsSessionsMiniBody');
            if (miniTableBody) {
                const history = data.sessions_history || [];
                if (!history.length) {
                    miniTableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted);">No sessions recorded yet.</td></tr>';
                } else {
                    miniTableBody.innerHTML = history.slice(0, 5).map(s => `
                        <tr>
                            <td><b>${escapeHtml(s.title)}</b><br><small style="color:var(--text-muted);">${new Date(s.start_time * 1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}</small></td>
                            <td style="color:${(s.avg_attention || 100) >= 75 ? '#10b981' : '#f59e0b'}; font-weight:700;">${(s.avg_attention || 100).toFixed(1)}%</td>
                            <td>${escapeHtml(s.dominant_expression || 'Neutral')}</td>
                            <td>
                                <button class="btn-mini-inspect" onclick="openSessionAnalysisModal('${s.session_uuid}')">Inspect</button>
                            </td>
                        </tr>
                    `).join('');
                }
            }

        } catch (e) {
            console.error('Error loading analytics trends:', e);
        }
    }

    // 4. Session Reports & Export
    const reportSessionSelect = document.getElementById('reportSessionSelect');
    const reportTerminal = document.getElementById('reportTerminal');
    const btnExportCsv = document.getElementById('btnExportCsv');
    const btnExportHtml = document.getElementById('btnExportHtml');

    async function loadReportsDropdown() {
        try {
            const res = await fetch('/api/sessions');
            if (!res.ok) return;
            const sessions = await res.json();
            reportSessionSelect.innerHTML = '<option value="">Select a session to inspect...</option>' +
                sessions.map(s => `<option value="${s.session_uuid}">${escapeHtml(s.title || 'Focus Session')} (${new Date(s.start_time * 1000).toLocaleTimeString()})</option>`).join('');
        } catch (e) {
            console.error(e);
        }
    }

    reportSessionSelect.addEventListener('change', () => {
        const uuid = reportSessionSelect.value;
        if (uuid) loadSelectedReport(uuid);
        else reportTerminal.textContent = 'Select a session above to generate summary report.';
    });

    async function loadSelectedReport(uuid) {
        try {
            reportTerminal.textContent = 'Generating authoritative session analytics...';
            const res = await fetch(`/api/reports/${uuid}/text`);
            if (res.ok) {
                const text = await res.text();
                reportTerminal.textContent = text;
            } else {
                reportTerminal.textContent = 'Error loading session report.';
            }
        } catch (e) {
            console.error(e);
        }
    }

    btnExportCsv.addEventListener('click', () => {
        const uuid = reportSessionSelect.value;
        if (!uuid) return alert('Please select a session first.');
        window.open(`/api/reports/${uuid}/csv`, '_blank');
    });

    btnExportHtml.addEventListener('click', () => {
        const uuid = reportSessionSelect.value;
        if (!uuid) return alert('Please select a session first.');
        window.open(`/api/reports/${uuid}/html`, '_blank');
    });

    // 5. ML Intelligence & Feature Importance
    const btnTrainMLModel = document.getElementById('btnTrainMLModel');
    const btnTestInference = document.getElementById('btnTestInference');
    const mlAccuracy = document.getElementById('mlAccuracy');
    const mlDatasetSize = document.getElementById('mlDatasetSize');
    const mlTrainSize = document.getElementById('mlTrainSize');
    const mlTestSize = document.getElementById('mlTestSize');
    const mlFeatureImportanceBars = document.getElementById('mlFeatureImportanceBars');
    const mlLivePrediction = document.getElementById('mlLivePrediction');

    async function loadMLInfo() {
        try {
            const res = await fetch('/api/ml/info');
            if (!res.ok) return;
            const data = await res.json();

            if (data.has_trained_model && data.metadata) {
                const meta = data.metadata;
                mlAccuracy.textContent = `${meta.accuracy}%`;
                mlDatasetSize.textContent = meta.total_samples;
                mlTrainSize.textContent = meta.train_samples;
                mlTestSize.textContent = meta.test_samples;

                if (meta.feature_importances) {
                    mlFeatureImportanceBars.innerHTML = Object.entries(meta.feature_importances)
                        .sort((a, b) => b[1] - a[1])
                        .map(([feat, imp]) => `
                            <div>
                                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#cbd5e1; margin-bottom:2px;">
                                    <span>${feat}</span>
                                    <b>${imp}%</b>
                                </div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill" style="width: ${imp}%; background: linear-gradient(90deg, #818cf8, #c084fc);"></div>
                                </div>
                            </div>
                        `).join('');
                }
            } else {
                mlAccuracy.textContent = '--';
                mlDatasetSize.textContent = '--';
            }
        } catch (e) {
            console.error(e);
        }
    }

    if (btnTrainMLModel) {
        btnTrainMLModel.addEventListener('click', async () => {
            btnTrainMLModel.textContent = '⏳ Training Random Forest Classifier...';
            try {
                const res = await fetch('/api/ml/train', { method: 'POST' });
                const result = await res.json();
                alert(result.message || 'Model trained!');
                loadMLInfo();
            } catch (e) {
                console.error(e);
            } finally {
                btnTrainMLModel.textContent = '⚡ Train / Re-train Model';
            }
        });
    }

    if (btnTestInference) {
        btnTestInference.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/ml/predict', { method: 'POST' });
                const pred = await res.json();
                mlLivePrediction.textContent = `${pred.predicted_label} (${pred.confidence}% Conf)`;
            } catch (e) {
                console.error(e);
            }
        });
    }

    function escapeHtml(str) {
        return str.replace(/[&<>'"]/g, tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag));
    }
});

