import { GameScene } from './GameScene.js';
import { MazeEnv } from './environments/MazeEnv.js';
import { QLearning } from './algorithms/QLearning.js';
import { DQN } from './algorithms/DQN.js';
import { PPO } from './algorithms/PPO.js';
import { CartPoleEnv } from './environments/CartPoleEnv.js';
const ENVS = { Maze: MazeEnv, CartPole: CartPoleEnv };
const ALGOS = { QL: QLearning, DQL: DQN, PPO: PPO };



const scene = new GameScene();
const config = {
    type: Phaser.AUTO,
    parent: 'game-container',
    width: 400,
    height: 400,
    scene: [scene],
    backgroundColor: '#1e293b'
};
const game = new Phaser.Game(config);
window.nudge = (target, dx, dy) => {
    // Prevent moving during training/playing
    if (scene.isTraining || scene.isPlaying) return;

    // Call the environment logic
    const success = scene.env.nudge(target, dx, dy);

    if (success) {
        // Update the coordinates in the UI text labels
        const pos = target === 'agent' ? scene.env.spawnPos : scene.env.goalPos;
        const label = document.getElementById(`pos-${target}`);
        if (label) label.innerText = `${pos.x},${pos.y}`;

        // Update the agent/goal in Phaser
        scene.sync();
    }
};
window.toggleMathPanel = function () {
    const panel = document.getElementById('math-panel');
    const arrow = document.getElementById('math-arrow');
    const isHidden = panel.classList.contains('hidden');

    if (isHidden) {
        panel.classList.remove('hidden');
        arrow.style.transform = 'rotate(180deg)';
        // Re-typeset math in case it wasn't rendered while hidden
        if (window.MathJax) {
            MathJax.typesetPromise();
        }
    } else {
        panel.classList.add('hidden');
        arrow.style.transform = 'rotate(0deg)';
    }
}
window.handleLoad = async function () {
    console.log("load called");
    const btn = document.getElementById('btnTrain');
    btn.innerText = "Loading Brain...";
    btn.disabled = true;

    // Use a tiny timeout to let the UI update the text before the heavy lifting starts
    await new Promise(r => setTimeout(r, 50));

    // Add these lines to debug
    console.log("Type of scene:", typeof scene);
    console.log("Value of scene:", scene);
    // End debug lines

    await scene.algo.loadModel();

    btn.innerText = "Brain Loaded!";

    // Reset the button after 2 seconds
    setTimeout(() => {
        btn.innerText = "START TRAINING";
        btn.disabled = false;
    }, 1000);
}

function algoConstrain(envKey) {
    const algoSelect = document.getElementById('algoSelect');
    const options = algoSelect.options;

    if (envKey === "CartPole") {
        // 1. UI Adjustments
        document.getElementById('btnRandom').style.display = "none";
        document.getElementById('trainEpisodes').value = 500; // PPO/DQL usually solve it by 500
        document.getElementById('trainSteps').value = 500;

        // 2. Enable/Disable specific algorithms
        // We enable PPO and DQL, but hide/disable QL (which is for Grids)
        for (let i = 0; i < options.length; i++) {
            const val = options[i].value;
            options[i].disabled = (val === "QL");
        }

        // 3. Set default and unlock the dropdown
        if (algoSelect.value === "QL") algoSelect.value = "PPO";
        algoSelect.disabled = false;

    } else {
        // 1. UI Adjustments for Maze/Grid
        document.getElementById('btnRandom').style.display = "block";
        document.getElementById('trainEpisodes').value = 100;
        document.getElementById('trainSteps').value = 200;

        // 2. Restrict to Q-Learning only
        for (let i = 0; i < options.length; i++) {
            const val = options[i].value;
            options[i].disabled = (val !== "QL");
        }

        // 3. Force selection and lock it
        algoSelect.value = "QL";
        algoSelect.disabled = true;
    }
}
/**
 * Dynamically builds the UI sliders and inputs based on 
 * the selected Environment and Algorithm metadata.
 */
function buildUI() {
    const envKey = document.getElementById('envSelect').value;
    const algoKey = document.getElementById('algoSelect').value;

    const envClass = ENVS[envKey];
    const algoClass = ALGOS[algoKey];
    console.log("key is:-" + algoKey)
    console.log("env is:- " + envClass);
    // 1. Build Reward Sliders (Environment Specific)
    const rewCont = document.getElementById('rewardContainer');
    rewCont.innerHTML = '';
    envClass.getMetadata().forEach(m => {
        rewCont.innerHTML += `
            <div class="flex flex-col gap-1">
                <div class="flex justify-between text-xs">
                    <span class="text-slate-400">${m.label}</span>
                </div>
                <input type="number" data-id="${m.id}" value="${m.default}" 
                       class="rew-input bg-slate-700 border border-slate-600 rounded px-2 py-1 text-sm outline-none focus:border-emerald-500">
            </div>`;
    });

    // 2. Build Hyperparameter Sliders (Algorithm Specific)
    const hypCont = document.getElementById('hyperContainer');
    hypCont.innerHTML = '';
    algoClass.getMetadata().forEach(m => {
        hypCont.innerHTML += `
            <div class="flex flex-col gap-1">
                <div class="flex justify-between text-xs">
                    <span class="text-slate-400">${m.label}</span>
                    <span id="v-${m.id}" class="text-emerald-400 font-mono">${m.default}</span>
                </div>
                <input type="range" data-id="${m.id}" min="${m.min}" max="${m.max}" step="${m.step}" value="${m.default}" 
                       class="hyp-input w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500">
            </div>`;
    });

    // Attach listeners to new inputs
    attachInputListeners();
    console.log(envKey);
    const envClassIns = new envClass();
    // Initialize the logic module in Phaser
    scene.initModule(envClassIns, new algoClass(envClassIns));

}
function buildAdvancedEnvUI(envClass) {
    const container = document.getElementById('envFieldsContainer');
    container.innerHTML = ''; // Clear previous env's UI

    const schema = envClass.getAdvancedConfigSchema();

    schema.forEach(item => {
        if (item.type === 'range') {
            container.innerHTML += `
            <div class="space-y-1 mb-4">
                <div class="flex justify-between text-[10px] text-slate-400 font-bold uppercase">
                    <span>${item.label}</span>
                    <span id="v-env-${item.id}">${scene.env[item.id]}</span>
                </div>
                <input 
                    type="range" 
                    data-id="${item.id}" 
                    min="${item.min}" 
                    max="${item.max}" 
                    step="${item.step}" 
                    value="${scene.env[item.id]}" 
                    class="env-config-range w-full h-1 bg-slate-700 accent-blue-500 rounded-lg appearance-none"
                >
            </div>`;
        }
        else if (item.type === 'dpad') {
            const colorClass = item.target === 'agent' ? 'text-emerald-400' : 'text-yellow-400';
            const pos = item.target === 'agent' ? scene.env.spawnPos : scene.env.goalPos;

            container.innerHTML += `
                <div class="bg-slate-900/40 p-3 rounded border border-slate-700 mb-4">
                    <span class="text-[10px] ${colorClass} font-bold uppercase">${item.label}</span>
                    <div class="flex justify-center mt-2">
                        <div class="grid grid-cols-3 gap-1">
                            <div></div>
                            <button onclick="window.nudge('${item.target}', 0, -1)" class="bg-slate-700 hover:bg-slate-600 p-2 rounded text-xs">▲</button>
                            <div></div>
                            <button onclick="window.nudge('${item.target}', -1, 0)" class="bg-slate-700 hover:bg-slate-600 p-2 rounded text-xs">◄</button>
                            <div class="flex items-center justify-center text-[10px] font-mono w-8" id="pos-${item.target}">${pos.x},${pos.y}</div>
                            <button onclick="window.nudge('${item.target}', 1, 0)" class="bg-slate-700 hover:bg-slate-600 p-2 rounded text-xs">►</button>
                            <div></div>
                            <button onclick="window.nudge('${item.target}', 0, 1)" class="bg-slate-700 hover:bg-slate-600 p-2 rounded text-xs">▼</button>
                            <div></div>
                        </div>
                    </div>
                </div>`;
        }
        else if (item.type === 'checkbox') {
            container.innerHTML += `
            <div class="flex items-center justify-between bg-slate-900/40 p-3 rounded border border-slate-700 mb-4 group hover:border-slate-500 transition-colors">
                <div class="flex flex-col">
                    <span class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">${item.label}</span>
                    <p class="text-[9px] text-slate-500 leading-tight pr-4">${item.description || ''}</p>
                </div>
                <label class="relative inline-flex items-center cursor-pointer">
                    <input 
                        type="checkbox" 
                        data-id="${item.id}"
                        class="env-config-checkbox sr-only peer"
                        ${item.default ? 'checked' : ''}
                    >
                    <div class="w-8 h-4 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
            </div>`;
        }
    });

    // Re-attach listeners for sliders
    document.querySelectorAll('.env-config-range, .env-config-checkbox').forEach(input => {
        input.oninput = (e) => {
            const target = e.target;
            const varName = target.dataset.id;

            // 1. Unified Value Extraction
            const val = target.type === 'checkbox' ? target.checked : parseFloat(target.value);
            scene.env[varName] = val;

            // 2. Range-only Label Update
            if (target.type === 'range') {
                document.getElementById(`v-env-${varName}`).innerText = val;
                scene.env.cols = val; // Legacy sync for Maze
            }

            // 3. Conditional Side Effects
            if (varName === 'length') {
                scene.pole.setSize(6, val * 200);
            }

            if (scene.env.name === "Maze world") {
                scene.randomizeAll();
            }
        };
    });
}
function attachInputListeners() {
    // Hyperparameter Listeners
    document.querySelectorAll('.hyp-input').forEach(input => {
        input.oninput = (e) => {
            const id = e.target.dataset.id;
            const val = parseFloat(e.target.value);
            document.getElementById(`v-${id}`).innerText = val;
            scene.algo.hypers[id] = val;
        };
    });

    // Reward Listeners
    document.querySelectorAll('.rew-input').forEach(input => {
        input.onchange = (e) => {
            const id = e.target.dataset.id;
            scene.env.params[id] = parseFloat(e.target.value);
        };
    });
}

// Function to handle routing and parameter extraction
async function handleRouting() {
    const queryString = window.location.search;
    const params = new URLSearchParams(queryString);
    if (params.has('env')) {
        console.log("handling");
        let envName = params.get('env');
        document.getElementById('envSelect').value = envName;
        console.log("env is" + ENVS[envName]);
        algoConstrain(envName);
        buildUI();
        buildAdvancedEnvUI(ENVS[envName]);
        if (envName === "CartPole") {
            await handleLoad();
        }
    } else {
        algoConstrain("Maze");
        document.getElementById('envSelect').value = "Maze";

        buildUI();
        buildAdvancedEnvUI(MazeEnv);
    }
}

function setUIState(state) {
    const isTraining = (state === 'TRAINING');
    const isPlaying = (state === 'PLAYING');
    const isBusy = (state === 'TRAINING' || state === 'PLAYING');
    document.getElementById('btnTrain').disabled = isTraining || isPlaying;
    document.getElementById('btnPlay').disabled = isTraining; // Disable Play while training
    document.getElementById('btnReset').disabled = isTraining || isPlaying;
    document.getElementById('btnRandom').disabled = isTraining || isPlaying;
    document.getElementById('envSelect').disabled = isTraining;

    // Visual feedback for the Train button
    const btnTrain = document.getElementById('btnTrain');
    btnTrain.classList.toggle('opacity-50', isTraining || isPlaying);
    btnTrain.classList.toggle('cursor-not-allowed', isTraining || isPlaying);

    // Visual feedback for the Sidebar to show it's "locked"
    const sidebar = document.querySelector('aside');
    if (isTraining) sidebar.classList.add('ring-2', 'ring-emerald-500/20');
    else sidebar.classList.remove('ring-2', 'ring-emerald-500/20');

    // Toggle Play/Stop Button Appearance
    const playBtn = document.getElementById('btnPlay');
    if (isPlaying) {
        playBtn.innerText = "Stop";
        playBtn.className = "w-full bg-red-600 hover:bg-red-500 py-2 rounded text-sm font-bold transition-colors";
    } else {
        playBtn.innerText = "Play";
        playBtn.className = "w-full bg-slate-700 hover:bg-slate-600 py-2 rounded text-sm transition-colors";
    }
}
const enableToggle = document.getElementById('enableVisToggle');
const speedContainer = document.getElementById('speedControlContainer');
const speedRadios = document.querySelectorAll('.speed-radio');

enableToggle.addEventListener('change', (e) => {
    const isEnabled = e.target.checked;

    if (isEnabled) {
        // Enable controls
        speedContainer.classList.remove('opacity-50', 'pointer-events-none');
        speedRadios.forEach(radio => radio.disabled = false);
    } else {
        // Disable controls
        speedContainer.classList.add('opacity-50', 'pointer-events-none');
        speedRadios.forEach(radio => radio.disabled = true);
    }
});
// Event Listeners
document.getElementById('btnTrain').onclick = async () => {
    if (scene) {
        console.log("Type of scene.algo:", typeof scene.algo);
        console.log("Value of scene.algo:", scene.algo);
        if (scene.algo) {
            console.log("Type of scene.algo.loadModel:", typeof scene.algo.loadModel);
        }
    }
    const episodes = parseInt(document.getElementById('trainEpisodes').value) || 100;
    const steps = parseInt(document.getElementById('trainSteps').value) || 200;
    // Lock UI and reset training interruption flag
    setUIState('TRAINING');

    // 1. IMPORTANT: We reset the "Brain" only when clicking "Start Training"
    // If the user interrupted a previous session, this starts a fresh brain.
    scene.algo.reset();
    scene.isInterrupting = false;

    document.getElementById('btnInterrupt').classList.remove('hidden');
    document.getElementById('progressBox').classList.remove('hidden');
    document.getElementById('progressHint').classList.remove('hidden');
    document.getElementById('progressBar').style.width = "0%"

    for (let i = 0; i < episodes; i++) {
        // 2. Check for interruption. 
        // If interrupted, the current Q-Table remains saved in scene.algo.qTable
        if (scene.isInterrupting) break;
        // 3. Run the episode. 
        // Note: We no longer pass 'shouldVisualize' because scene.runEpisode 
        // now checks the Radio button value live on every step.
        await scene.runEpisode(steps, enableToggle.checked);

        // Update UI Progress Bar
        document.getElementById('progressBar').style.width = `${((i + 1) / episodes) * 100}%`;
    }
    if (scene.env.name === "CartPole") await scene.algo.saveModel();
    // 4. Return UI to normal state
    setUIState('IDLE');
    document.getElementById('btnInterrupt').classList.add('hidden');
    document.getElementById('progressBox').classList.add('hidden');
    document.getElementById('progressHint').classList.add('hidden');

    // Return agent to the "Home" spawn point after training finishes or is stopped
    scene.resetToHome();
};
document.getElementById('btnPlay').onclick = () => {
    if (scene.isPlaying) {
        scene.isPlaying = false;
        scene.resetToHome();
        setUIState('IDLE');
    } else {
        scene.result = { done: false };
        scene.isPlaying = true;
        setUIState('PLAYING');
    }
};

document.getElementById('btnReset').onclick = () => scene.resetToRandom();
document.getElementById('btnRandom').onclick = () => scene.randomizeAll();
document.getElementById('btnInterrupt').onclick = () => scene.isInterrupting = true;
document.getElementById('algoSelect').onchange = buildUI;
document.getElementById('envSelect').onchange = async (e) => {
    window.location.href = `index.html?env=${e.target.value}`

};
document.addEventListener('DOMContentLoaded', async (e) => {
    await handleRouting();
});
