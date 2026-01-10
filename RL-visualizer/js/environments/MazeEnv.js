export class MazeEnv {
    constructor(rows = 10, cols = 10) {
        this.name = "Maze world";
        this.rows = rows;
        this.cols = cols;
        this.grid = null;
        this.agentPos = { x: 0, y: 0 };
        this.goalPos = { x: 0, y: 0 };
        this.spawnPos = { x: 0, y: 0 }; // The "Home" position
        this.params = { goal: 100, survival: -1, damage: -10 };
        this.randomize();
    }

    static getMetadata() {
        return [
            { id: 'goal', label: 'Goal Reward', default: 100 },
            { id: 'survival', label: 'Survival Penalty', default: -1 },
            { id: 'damage', label: 'Wall Penalty', default: -10 }
        ];
    }
    static getAdvancedConfigSchema() {
        return [
            { id: 'rows', label: 'Grid size', type: 'range', min: 4, max: 20, step: 1, default: 10 },
            { id: 'agent-spawn', label: 'Agent Spawn', type: 'dpad', target: 'agent' },
            { id: 'goal-spawn', label: 'Goal Position', type: 'dpad', target: 'goal' }
        ];
    }
    nudge(target, dx, dy) {
        const obj = (target === 'agent') ? this.spawnPos : this.goalPos;
        const newX = Math.max(0, Math.min(this.cols - 1, obj.x + dx));
        const newY = Math.max(0, Math.min(this.rows - 1, obj.y + dy));

        if (this.grid[newY][newX] === 0) {
            obj.x = newX;
            obj.y = newY;
            if (target === 'agent') this.agentPos = { ...this.spawnPos };

            return true;
        }
        return false;
    }
    randomize() {
        this.grid = Array.from({ length: this.rows }, () =>
            Array.from({ length: this.cols }, () => (Math.random() < 0.15 ? 1 : 0))
        );
        this.goalPos = this.getRandomEmpty();
        this.spawnPos = this.getRandomEmpty();
        this.agentPos = { ...this.spawnPos };
    }

    // New logic: if goToHome is true, return to spawn. Else, pick new random spot.
    reset(goToHome = false) {
        if (goToHome) {
            this.agentPos = { ...this.spawnPos };
        } else {
            this.agentPos = this.getRandomEmpty();
            this.spawnPos = this.agentPos;
        }
        return [this.agentPos.x, this.agentPos.y, this.goalPos.x, this.goalPos.y];
    }

    step(action) {
        let reward = this.params.survival;
        let nx = this.agentPos.x, ny = this.agentPos.y;
        if (action === 0) ny--; if (action === 1) ny++; if (action === 2) nx--; if (action === 3) nx++;

        if (nx < 0 || nx >= this.cols || ny < 0 || ny >= this.rows || this.grid[ny][nx] === 1) {
            reward += this.params.damage;
        } else {
            this.agentPos = { x: nx, y: ny };
        }
        const distBefore = Math.abs(nx - this.goalPos.x) + Math.abs(ny - this.goalPos.y);
        const distAfter = Math.abs(this.agentPos.x - this.goalPos.x) + Math.abs(this.agentPos.y - this.goalPos.y);

        if (distAfter < distBefore) reward += 0.1;
        const done = (this.agentPos.x === this.goalPos.x && this.agentPos.y === this.goalPos.y);
        if (done) reward += this.params.goal;
        return { obs: [this.agentPos.x, this.agentPos.y, this.goalPos.x, this.goalPos.y], reward, done };
    }

    getRandomEmpty() {
        let x, y;
        do {
            x = Math.floor(Math.random() * this.cols);
            y = Math.floor(Math.random() * this.rows);
        } while (this.grid[y][x] === 1 || (this.goalPos && x === this.goalPos.x && y === this.goalPos.y));
        return { x, y };
    }
}