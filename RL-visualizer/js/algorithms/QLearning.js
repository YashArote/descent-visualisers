export class QLearning {
    constructor() {
        this.reset();
        this.hypers = { epsilon: 0.1, alpha: 0.1, gamma: 0.9 };
    }

    // Wipe the brain
    reset() {
        this.qTable = {};
    }

    static getMetadata() {
        return [
            { id: 'epsilon', label: 'Curiosity (ε)', min: 0, max: 1, step: 0.01, default: 0.1 },
            { id: 'alpha', label: 'Learning Rate (α)', min: 0.01, max: 1, step: 0.01, default: 0.1 },
            { id: 'gamma', label: 'Future Bias (γ)', min: 0, max: 1, step: 0.01, default: 0.9 }
        ];
    }

    getQ(s) {
        const k = s.join(',');
        if (!this.qTable[k]) this.qTable[k] = [0, 0, 0, 0];
        return this.qTable[k];
    }

    act(s, forceDeterministic = false) {
        if (Math.random() < this.hypers.epsilon) return Math.floor(Math.random() * 4);
        const qs = this.getQ(s);
        return qs.indexOf(Math.max(...qs));
    }

    learn(s, a, r, sn, done) {
        const qs = this.getQ(s);
        const nextMax = done ? 0 : Math.max(...this.getQ(sn));
        qs[a] += this.hypers.alpha * (r + this.hypers.gamma * nextMax - qs[a]);
    }
}