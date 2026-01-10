export class GameScene extends Phaser.Scene {
    constructor() {
        super("GameScene");
        this.isInterrupting = false;
        this.isPlaying = false;
        this.accumulatedTime = 0;
        this.result = { done: false };
    }

    initModule(env, algo) {
        console.log("inside init");
        this.env = env;
        this.algo = algo;
        if (this.add) { // If Phaser is already running
            this.createVisuals(); // Wipes the old and draws the new
        }
        // Draw immediately if the scene is already active

    }
    restart() {
        this.scene.restart();
    }
    create() {
        // Runs once Phaser boots up
        if (this.env) this.createVisuals();
    }

    createVisuals() {
        if (!this.env) return;

        // 1. Clear previous objects
        this.children.removeAll();
        console.log("outside cartpole")
        if (this.env.name === "CartPole") {
            console.log("inside cartpole")
            this.add.rectangle(200, 300, 400, 2, 0x64748b);
            this.cart = this.add.rectangle(200, 300, 60, 30, 0x10b981).setOrigin(0.5);
            this.pole = this.add.rectangle(200, 285, 6, 100, 0xfacc15).setOrigin(0.5, 1);
        } else {
            console.log("inside Maze")
            const containerSize = 400;
            const padding = 20;
            const availableSpace = containerSize - padding;
            this.tileSize = availableSpace / this.env.cols;
            this.gridOffset = {
                x: (containerSize - (this.env.cols * this.tileSize)) / 2,
                y: (containerSize - (this.env.rows * this.tileSize)) / 2
            };

            // 5. Draw the Tiles
            for (let y = 0; y < this.env.rows; y++) {
                for (let x = 0; x < this.env.cols; x++) {
                    const isWall = this.env.grid[y][x] === 1;
                    const color = isWall ? 0x334155 : 0x1e293b;

                    this.add.rectangle(
                        this.gridOffset.x + (x * this.tileSize),
                        this.gridOffset.y + (y * this.tileSize),
                        this.tileSize - 1,
                        this.tileSize - 1,
                        color
                    ).setOrigin(0);
                }
            }

            // 6. Scale Agent/Goal dynamically
            // A base scale of 1.0 corresponds to a 10x10 grid (tileSize ~38)
            const scaleFactor = this.tileSize / 38;

            this.goal = this.add.star(0, 0, 5, 8 * scaleFactor, 15 * scaleFactor, 0xfacc15).setOrigin(0.5);
            this.agent = this.add.circle(0, 0, 12 * scaleFactor, 0x10b981).setOrigin(0.5);
        }
        this.sync();
    }

    sync(isTraining = false) {
        if (this.env.name === "CartPole") {
            const [x, xDot, theta, thetaDot] = this.env.state;
            // Map physics X (-2.4 to 2.4) to screen (50 to 350)
            const screenX = 200 + (x / this.env.xThreshold) * 180;

            this.cart.x = screenX;
            this.pole.x = screenX;
            this.pole.rotation = theta; // Theta is in radians
        } else {
            if (!this.agent || !this.goal || !this.gridOffset) return;

            // Math: Offset + (Coordinate * TileSize) + (Half of TileSize to Center)
            const agentX = this.gridOffset.x + (this.env.agentPos.x * this.tileSize) + (this.tileSize / 2);
            const agentY = this.gridOffset.y + (this.env.agentPos.y * this.tileSize) + (this.tileSize / 2);

            const goalX = this.gridOffset.x + (this.env.goalPos.x * this.tileSize) + (this.tileSize / 2);
            const goalY = this.gridOffset.y + (this.env.goalPos.y * this.tileSize) + (this.tileSize / 2);
            if (!isTraining) {
                this.agent.setPosition(agentX, agentY);
                this.goal.setPosition(goalX, goalY);
            }
        }
    }

    resetToRandom() {
        this.env.reset(false);
        this.sync();
    }

    resetToHome() {
        this.env.reset(true);
        this.sync();
    }

    randomizeAll() {
        this.env.randomize();
        this.createVisuals();
    }

    /**
     * Main Training Episode Loop
     * @param {boolean} visualize - If true, we show every step slowly
     */
    async runEpisode(trainSteps, toVisualize) {
        // 1. Get raw state
        let rawState = this.env.reset(false);

        // 2. NORMALIZE IT IMMEDIATELY (The Fix) 🛠️
        let state = rawState;

        let done = false;
        let steps = 0;

        while (!done && !this.isInterrupting && steps < trainSteps) {
            const action = this.algo.act(state);
            const { obs, reward, done: d } = this.env.step(action);
            await this.algo.learn(state, action, reward, obs, d);

            state = obs; 
            done = d;
            steps++;

            const speed = document.querySelector('input[name="trainSpeed"]:checked').value;
            if (toVisualize) {
                if (speed === 'slow') {
                    this.sync();
                    await new Promise(r => setTimeout(r, 50));
                } else {

                    if (this.env.name === "CartPole") {
                        await new Promise(r => setTimeout(r, 0));
                    } else {
                        await new Promise(r => setTimeout(r, 20));
                    }
                    this.sync();
                }
            }
        }

    }

    update(time, delta) {
        if (this.isPlaying) {
            this.accumulatedTime += delta;
            const tickRate = this.env.name === "CartPole" ? 20 : 150;

            if (this.accumulatedTime > tickRate) {
                if (this.result.done) {
                    this.resetToHome();
                    const playBtn = document.getElementById('btnPlay');
                    if (playBtn) playBtn.click();
                }

                
                let currentState = this.env.state || [this.env.agentPos.x, this.env.agentPos.y, this.env.goalPos.x, this.env.goalPos.y];

                if (this.env.name === "CartPole") {
                    currentState = [
                        currentState[0] / this.env.xThreshold,
                        currentState[1] / 10,
                        currentState[2] / this.env.thetaThresholdRadians,
                        currentState[3] / 10
                    ];
                }
                

               
                const action = this.algo.act(currentState, true);
                this.result = this.env.step(action);
                this.sync();

                this.accumulatedTime = 0;
            }
        }
    }
}