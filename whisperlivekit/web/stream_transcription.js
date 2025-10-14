const isExtension = typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.getURL;
if (isExtension) {
    document.documentElement.classList.add('is-extension');
}

let isTranscribing = false;
let websocket = null;
let websocketUrl = "ws://localhost:8000/asr";
let userClosing = false;
let wakeLock = null;
let startTime = null;
let timerInterval = null;
let waitingForStop = false;
let lastReceivedData = null;
let lastSignature = null;
let selectedLanguage = "auto";
let autoScrollEnabled = true;
const LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
};

const statusText = document.getElementById("status");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const urlInput = document.getElementById("urlInput");
const linesTranscriptDiv = document.getElementById("linesTranscript");
const timerElement = document.querySelector(".timer");
const themeRadios = document.querySelectorAll('input[name="theme"]');
const languageSelect = document.getElementById("languageSelect");

const settingsDiv = document.querySelector(".settings");

const translationIcon = `<svg xmlns="http://www.w3.org/2000/svg" height="12px" viewBox="0 -960 960 960" width="12px" fill="#5f6368"><path d="m603-202-34 97q-4 11-14 18t-22 7q-20 0-32.5-16.5T496-133l152-402q5-11 15-18t22-7h30q12 0 22 7t15 18l152 403q8 19-4 35.5T868-80q-13 0-22.5-7T831-106l-34-96H603ZM362-401 188-228q-11 11-27.5 11.5T132-228q-11-11-11-28t11-28l174-174q-35-35-63.5-80T190-640h84q20 39 40 68t48 58q33-33 68.5-92.5T484-720H80q-17 0-28.5-11.5T40-760q0-17 11.5-28.5T80-800h240v-40q0-17 11.5-28.5T360-880q17 0 28.5 11.5T400-840v40h240q17 0 28.5 11.5T680-760q0 17-11.5 28.5T640-720h-76q-21 72-63 148t-83 116l96 98-30 82-122-125Zm266 129h144l-72-204-72 204Z"/></svg>`
const silenceIcon = `<svg xmlns="http://www.w3.org/2000/svg" style="vertical-align: text-bottom;" height="14px" viewBox="0 -960 960 960" width="14px" fill="#5f6368"><path d="M514-556 320-752q9-3 19-5.5t21-2.5q66 0 113 47t47 113q0 11-1.5 22t-4.5 22ZM40-200v-32q0-33 17-62t47-44q51-26 115-44t141-18q26 0 49.5 2.5T456-392l-56-54q-9 3-19 4.5t-21 1.5q-66 0-113-47t-47-113q0-11 1.5-21t4.5-19L84-764q-11-11-11-28t11-28q12-12 28.5-12t27.5 12l675 685q11 11 11.5 27.5T816-80q-11 13-28 12.5T759-80L641-200h39q0 33-23.5 56.5T600-120H120q-33 0-56.5-23.5T40-200Zm80 0h480v-32q0-14-4.5-19.5T580-266q-36-18-92.5-36T360-320q-71 0-127.5 18T140-266q-9 5-14.5 14t-5.5 20v32Zm240 0Zm560-400q0 69-24.5 131.5T829-355q-12 14-30 15t-32-13q-13-13-12-31t12-33q30-38 46.5-85t16.5-98q0-51-16.5-97T767-781q-12-15-12.5-33t12.5-32q13-14 31.5-13.5T829-845q42 51 66.5 113.5T920-600Zm-182 0q0 32-10 61.5T700-484q-11 15-29.5 15.5T638-482q-13-13-13.5-31.5T633-549q6-11 9.5-24t3.5-27q0-14-3.5-27t-9.5-25q-9-17-8.5-35t13.5-31q14-14 32.5-13.5T700-716q18 25 28 54.5t10 61.5Z"/></svg>`;
const languageIcon = `<svg xmlns="http://www.w3.org/2000/svg" height="12" viewBox="0 -960 960 960" width="12" fill="#5f6368"><path d="M480-80q-82 0-155-31.5t-127.5-86Q143-252 111.5-325T80-480q0-83 31.5-155.5t86-127Q252-817 325-848.5T480-880q83 0 155.5 31.5t127 86q54.5 54.5 86 127T880-480q0 82-31.5 155t-86 127.5q-54.5 54.5-127 86T480-80Zm0-82q26-36 45-75t31-83H404q12 44 31 83t45 75Zm-104-16q-18-33-31.5-68.5T322-320H204q29 50 72.5 87t99.5 55Zm208 0q56-18 99.5-55t72.5-87H638q-9 38-22.5 73.5T584-178ZM170-400h136q-3-20-4.5-39.5T300-480q0-21 1.5-40.5T306-560H170q-5 20-7.5 39.5T160-480q0 21 2.5 40.5T170-400Zm216 0h188q3-20 4.5-39.5T580-480q0-21-1.5-40.5T574-560H386q-3 20-4.5 39.5T380-480q0 21 1.5 40.5T386-400Zm268 0h136q5-20 7.5-39.5T800-480q0-21-2.5-40.5T790-560H654q3 20 4.5 39.5T660-480q0 21-1.5 40.5T654-400Zm-16-240h118q-29-50-72.5-87T584-782q18 33 31.5 68.5T638-640Zm-234 0h152q-12-44-31-83t-45-75q-26 36-45 75t-31 83Zm-200 0h118q9-38 22.5-73.5T376-782q-56 18-99.5 55T204-640Z"/></svg>`
const speakerIcon = `<svg xmlns="http://www.w3.org/2000/svg" height="16px" style="vertical-align: text-bottom;" viewBox="0 -960 960 960" width="16px" fill="#5f6368"><path d="M480-480q-66 0-113-47t-47-113q0-66 47-113t113-47q66 0 113 47t47 113q0 66-47 113t-113 47ZM160-240v-32q0-34 17.5-62.5T224-378q62-31 126-46.5T480-440q66 0 130 15.5T736-378q29 15 46.5 43.5T800-272v32q0 33-23.5 56.5T720-160H240q-33 0-56.5-23.5T160-240Zm80 0h480v-32q0-11-5.5-20T700-306q-54-27-109-40.5T480-360q-56 0-111 13.5T260-306q-9 5-14.5 14t-5.5 20v32Zm240-320q33 0 56.5-23.5T560-640q0-33-23.5-56.5T480-720q-33 0-56.5 23.5T400-640q0 33 23.5 56.5T480-560Zm0-80Zm0 400Z"/></svg>`;

function applyTheme(pref) {
    if (pref === "light") {
        document.documentElement.setAttribute("data-theme", "light");
    } else if (pref === "dark") {
        document.documentElement.setAttribute("data-theme", "dark");
    } else {
        document.documentElement.removeAttribute("data-theme");
    }
}

// Persisted theme preference
const savedThemePref = localStorage.getItem("themePreference") || "system";
applyTheme(savedThemePref);
if (themeRadios.length) {
    themeRadios.forEach((r) => {
        r.checked = r.value === savedThemePref;
        r.addEventListener("change", () => {
            if (r.checked) {
                localStorage.setItem("themePreference", r.value);
                applyTheme(r.value);
            }
        });
    });
}

function populateLanguageSelect() {
    if (!languageSelect) return;

    for (const [code, name] of Object.entries(LANGUAGES)) {
        const option = document.createElement('option');
        option.value = code;
        option.textContent = name.charAt(0).toUpperCase() + name.slice(1);
        languageSelect.appendChild(option);
    }

    const savedLanguage = localStorage.getItem('selectedLanguage');
    if (savedLanguage && LANGUAGES[savedLanguage]) {
        languageSelect.value = savedLanguage;
        selectedLanguage = savedLanguage;
    } else {
        languageSelect.value = "auto";
        selectedLanguage = "auto";
    }
}

function handleLanguageChange() {
    selectedLanguage = languageSelect.value;
    localStorage.setItem('selectedLanguage', selectedLanguage);
    statusText.textContent = `Язык изменен на: ${selectedLanguage}`;

    if (isTranscribing) {
        statusText.textContent = "Переключение языка... Пожалуйста подождите.";
        stopTranscription().then(() => {
            setTimeout(() => {
                startTranscription();
            }, 1000);
        });
    }
}

// Helpers
function fmt1(x) {
    const n = Number(x);
    return Number.isFinite(n) ? n.toFixed(1) : x;
}

let host, port, protocol;

if (isExtension) {
    host = "localhost";
    port = 8000;
    protocol = "ws";
} else {
    host = window.location.hostname || "localhost";
    port = window.location.port || (window.location.protocol === "https:" ? "" : 8000);
    protocol = window.location.protocol === "https:" ? "wss" : "ws";
}
const defaultWebSocketUrl = `${protocol}://${host}${port ? ":" + port : ""}/asr`;

websocketUrl = defaultWebSocketUrl;

function setupWebSocket() {
    return new Promise((resolve, reject) => {
        try {
            const urlToTranscribe = urlInput.value.trim();
            if (!urlToTranscribe) {
                statusText.textContent = "Пожалуйста, введите URL для транскрипции.";
                reject("Не указан URL");
                return;
            }

            const currentUrl = new URL(websocketUrl);
            currentUrl.searchParams.set("language", selectedLanguage);
            currentUrl.searchParams.set("url", urlToTranscribe);
            websocket = new WebSocket(currentUrl.toString());
        } catch (error) {
            statusText.textContent = "Неверный WebSocket URL. Пожалуйста, проверьте и попробуйте снова.";
            reject(error);
            return;
        }

        websocket.onopen = () => {
            statusText.textContent = "Подключено к серверу. Транскрипция запущена.";
            resolve();
        };

        websocket.onclose = () => {
            if (userClosing) {
                if (waitingForStop) {
                    statusText.textContent = "Обработка завершена или соединение закрыто.";
                    if (lastReceivedData) {
                        renderLinesWithBuffer(
                            lastReceivedData.lines || [],
                            lastReceivedData.buffer_diarization || "",
                            lastReceivedData.buffer_transcription || "",
                            0,
                            0,
                            true
                        );
                    }
                }
            } else {
                statusText.textContent = "Отключено от WebSocket сервера.";
                if (isTranscribing) {
                    stopTranscription();
                }
            }
            isTranscribing = false;
            waitingForStop = false;
            userClosing = false;
            lastReceivedData = null;
            websocket = null;
            updateUI();
        };

        websocket.onerror = () => {
            statusText.textContent = "Ошибка подключения к WebSocket.";
            reject(new Error("Ошибка подключения к WebSocket"));
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === "ready_to_stop") {
                console.log("Ready to stop received, finalizing display and closing WebSocket.");
                waitingForStop = false;

                if (lastReceivedData) {
                    renderLinesWithBuffer(
                        lastReceivedData.lines || [],
                        lastReceivedData.buffer_diarization || "",
                        lastReceivedData.buffer_transcription || "",
                        0,
                        0,
                        true
                    );
                }
                statusText.textContent = "Обработка аудио завершена! Готов к новой транскрипции.";
                updateUI();

                if (websocket) {
                    websocket.close();
                }
                return;
            }

            lastReceivedData = data;

            const {
                lines = [],
                buffer_transcription = "",
                buffer_diarization = "",
                remaining_time_transcription = 0,
                remaining_time_diarization = 0,
                status = "active_transcription",
            } = data;

            renderLinesWithBuffer(
                lines,
                buffer_diarization,
                buffer_transcription,
                remaining_time_diarization,
                remaining_time_transcription,
                false,
                status
            );
        };
    });
}

function renderLinesWithBuffer(
    lines,
    buffer_diarization,
    buffer_transcription,
    remaining_time_diarization,
    remaining_time_transcription,
    isFinalizing = false,
    current_status = "active_transcription"
) {
    if (current_status === "no_audio_detected") {
        linesTranscriptDiv.innerHTML =
            "<p style='text-align: center; color: var(--muted); margin-top: 20px;'><em>Аудио не обнаружено...</em></p>";
        return;
    }

    const showLoading = !isFinalizing && (lines || []).some((it) => it.speaker == 0);
    const showTransLag = !isFinalizing && remaining_time_transcription > 0;
    const showDiaLag = !isFinalizing && !!buffer_diarization && remaining_time_diarization > 0;
    const signature = JSON.stringify({
        lines: (lines || []).map((it) => ({
            speaker: it.speaker,
            text: it.text,
            start: it.start,
            end: it.end,
            detected_language: it.detected_language
        })),
        buffer_transcription: buffer_transcription || "",
        buffer_diarization: buffer_diarization || "",
        status: current_status,
        showLoading,
        showTransLag,
        showDiaLag,
        isFinalizing: !!isFinalizing,
    });
    if (lastSignature === signature) {
        const t = document.querySelector(".lag-transcription-value");
        if (t) t.textContent = fmt1(remaining_time_transcription);
        const d = document.querySelector(".lag-diarization-value");
        if (d) d.textContent = fmt1(remaining_time_diarization);
        const ld = document.querySelector(".loading-diarization-value");
        if (ld) ld.textContent = fmt1(remaining_time_diarization);
        return;
    }
    lastSignature = signature;

    const linesHtml = (lines || [])
        .map((item, idx) => {
            let timeInfo = "";
            if (item.start !== undefined && item.end !== undefined) {
                timeInfo = ` ${item.start} - ${item.end}`;
            }

            let speakerLabel = "";
            if (item.speaker === -2) {
                //console.log(`Обнаружена тишина ${timeInfo}`)
                //speakerLabel = `<span class="silence">${silenceIcon}<span id='timeInfo'>${timeInfo}</span></span>`;
            } else if (item.speaker == 0 && !isFinalizing) {
                speakerLabel = `<span class='loading'><span class="spinner"></span><span class="loading-diarization-value">${fmt1(
                    remaining_time_diarization
                )}</span> сек. аудио обрабатывается диаризацией</span></span>`;
            } else if (item.speaker !== 0) {
                const speakerNum = `<span class="speaker-badge">${item.speaker}</span>`;
                //speakerLabel = `<span id="speaker">${speakerIcon}${speakerNum}</span>`;

                if (item.detected_language) {
                    speakerLabel += `<span class="label_language">${languageIcon}<span>${item.detected_language}</span></span>`;
                }
            }

            let currentLineText = item.text || "";

            if (idx === lines.length - 1) {
                if (!isFinalizing && item.speaker !== -2) {
                    if (remaining_time_transcription > 0) {
                        speakerLabel += `<span class="label_transcription"><span class="spinner"></span>Задержка транскрипции <span id='timeInfo'><span class="lag-transcription-value">${fmt1(
                            remaining_time_transcription
                        )}</span>с</span></span>`;
                    }
                    if (buffer_diarization && remaining_time_diarization > 0) {
                        speakerLabel += `<span class="label_diarization"><span class="spinner"></span>Задержка диаризации<span id='timeInfo'><span class="lag-diarization-value">${fmt1(
                            remaining_time_diarization
                        )}</span>с</span></span>`;
                    }
                }

                if (buffer_diarization) {
                    if (isFinalizing) {
                        currentLineText +=
                            (currentLineText.length > 0 && buffer_diarization.trim().length > 0 ? " " : "") + buffer_diarization.trim();
                    } else {
                        currentLineText += `<span class="buffer_diarization">${buffer_diarization}</span>`;
                    }
                }
                if (buffer_transcription) {
                    if (isFinalizing) {
                        currentLineText +=
                            (currentLineText.length > 0 && buffer_transcription.trim().length > 0 ? " " : "") +
                            buffer_transcription.trim();
                    } else {
                        currentLineText += `<span class="buffer_transcription">${buffer_transcription}</span>`;
                    }
                }
            }

            if (item.translation) {
                currentLineText += `
            <div>
                <div class="label_translation">
                    ${translationIcon}
                    <span>${item.translation}</span>
                </div>
            </div>`;
            }

            return currentLineText.trim().length > 0 || speakerLabel.length > 0
                ? `<p>${speakerLabel}<br/><div class='textcontent'>${currentLineText}</div></p>`
                : `<p>${speakerLabel}<br/></p>`;
        })
        .join("");

    linesTranscriptDiv.innerHTML = linesHtml;
    const transcriptContainer = document.querySelector('.transcript-container');
    if (transcriptContainer && autoScrollEnabled) {
        transcriptContainer.scrollTo({top: transcriptContainer.scrollHeight, behavior: "smooth"});
    }
}

function updateTimer() {
    if (!startTime) return;

    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, "0");
    const seconds = (elapsed % 60).toString().padStart(2, "0");
    timerElement.textContent = `${minutes}:${seconds}`;
}

async function startTranscription() {
    if (isTranscribing || waitingForStop) return;

    console.log("Connecting to WebSocket");
    try {
        await setupWebSocket();
        
        try {
            wakeLock = await navigator.wakeLock.request("screen");
        } catch (err) {
            console.log("Error acquiring wake lock.");
        }

        startTime = Date.now();
        timerInterval = setInterval(updateTimer, 1000);

        isTranscribing = true;
        updateUI();
    } catch (err) {
        statusText.textContent = "Не удалось подключиться к WebSocket. Прервано.";
        console.error(err);
        updateUI(); // Ensure UI is consistent on failure
    }
}

async function stopTranscription() {
    if (wakeLock) {
        try {
            await wakeLock.release();
        } catch (e) {
            // ignore
        }
        wakeLock = null;
    }

    userClosing = true;
    waitingForStop = true;

    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.close();
        statusText.textContent = "Транскрипция остановлена. Обрабатывается финальное аудио...";
    }

    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
    timerElement.textContent = "00:00";
    startTime = null;

    isTranscribing = false;
    updateUI();
}

function updateUI() {
    startButton.disabled = isTranscribing || waitingForStop;
    stopButton.disabled = !isTranscribing || waitingForStop;

    if (waitingForStop) {
        if (statusText.textContent !== "Транскрипция остановлена. Обрабатывается финальное аудио...") {
            statusText.textContent = "Пожалуйста, подождите завершения обработки...";
        }
    } else if (isTranscribing) {
        statusText.textContent = "";
    } else {
        if (
            statusText.textContent !== "Обработка аудио завершена! Готов к новой транскрипции." &&
            statusText.textContent !== "Обработка завершена или соединение закрыто."
        ) {
            statusText.textContent = "Нажмите 'Запуск' для начала транскрипции";
        }
    }
}

startButton.addEventListener("click", startTranscription);
stopButton.addEventListener("click", stopTranscription);

if (languageSelect) {
    languageSelect.addEventListener("change", handleLanguageChange);
}
document.addEventListener('DOMContentLoaded', async () => {
    populateLanguageSelect();
    updateUI(); // Set initial button states

    // Setup autoscroll control
    const transcriptContainer = document.querySelector('.transcript-container');
    if (transcriptContainer) {
        transcriptContainer.addEventListener('scroll', () => {
            const { scrollTop, scrollHeight, clientHeight } = transcriptContainer;
            const isScrolledToBottom = scrollHeight - scrollTop - clientHeight < 50;

            if (isScrolledToBottom) {
                autoScrollEnabled = true;
            } else {
                autoScrollEnabled = false;
            }
        });
    }
});
