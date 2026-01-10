export class CartPoleEnv {
    constructor() {
        this.name = "CartPole";
        // Physics constants
        this.gravity = 9.8;
        this.massCart = 1.0;
        this.massPole = 0.1;
        this.totalMass = this.massPole + this.massCart;
        this.length = 0.5; // half pole length
        this.poleMassLength = this.massPole * this.length;
        this.forceMag = 10.0;
        this.tau = 0.02; // seconds between state updates
        this.randomStart = true;
        this.xThreshold = 2.4;
        this.thetaThresholdRadians = 12 * 2 * Math.PI / 360;

        // Reward Parameters (Mapped to getMetadata)
        this.params = {
            survival: 1,
            fallPenalty: -50,
            centerBias: 0.8
        };

        this.reset();
    }

    static getAdvancedConfigSchema() {
        return [
            { id: 'gravity', label: 'Gravity', type: 'range', min: 1, max: 20, step: 0.1, default: 9.8 },
            { id: 'length', label: 'Pole Length', type: 'range', min: 0.1, max: 2, step: 0.1, default: 0.5 },
            { id: 'forceMag', label: 'Push Force', type: 'range', min: 1, max: 30, step: 1, default: 10.0 },
            // Added Checkbox
            {
                id: 'randomStart',
                label: 'Randomized Start',
                type: 'checkbox',
                default: false,
                description: 'Starts the cart at random positions other than center.'
            }
        ];
    }

    static getMetadata() {
        return [
            { id: 'survival', label: 'Survival Reward', default: 1 },
            { id: 'fallPenalty', label: 'Fall Penalty', default: -50 },
            { id: 'centerBias', label: 'Center Bias', min: 0, max: 2.0, step: 0.1, default: 0.8 }
        ];
    }

    getNormalizedState() {
        const [x, xDot, theta, thetaDot] = this.state;
        return [
            x / this.xThreshold,
            xDot / 10,
            theta / this.thetaThresholdRadians,
            thetaDot / 10
        ];
    }

    reset(random = true) {
        // If 'random' is true, start anywhere on screen. 
        // If false (default), start near center.
        const xStart = this.randomStart ? (Math.random() - 0.5) * 2.0 : (Math.random() - 0.5) * 0.1;

        this.state = [
            xStart,
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1
        ];
        this.steps = 0;
        return this.getNormalizedState();
    }

    step(action) {
        let [x, xDot, theta, thetaDot] = this.state;
        const force = action === 1 ? this.forceMag : -this.forceMag;
        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);

        const temp = (force + this.poleMassLength * thetaDot * thetaDot * sinTheta) / this.totalMass;
        const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) / (this.length * (4.0 / 3.0 - this.massPole * cosTheta * cosTheta / this.totalMass));
        const xAcc = temp - this.poleMassLength * thetaAcc * cosTheta / this.totalMass;

        x += this.tau * xDot;
        xDot += this.tau * xAcc;
        theta += this.tau * thetaDot;
        thetaDot += this.tau * thetaAcc;

        this.state = [x, xDot, theta, thetaDot];

        const outOfBounds = x < -this.xThreshold || x > this.xThreshold ||
            theta < -this.thetaThresholdRadians || theta > this.thetaThresholdRadians;

        const done = outOfBounds;
        this.steps++;

        // --- NEW "SWING BACK" REWARD LOGIC ---
        let reward = 0;

        if (done) {
            reward = this.params.fallPenalty; // -50 (Death is the worst)
        } else {
            reward = this.params.survival;
            const distRatio = Math.abs(x) / this.xThreshold;
            const centerBias = this.params.centerBias || 0.8;
            reward -= centerBias * Math.pow(distRatio, 2);
            const moveAwayPenalty = (x * xDot);
            reward -= moveAwayPenalty * 0.1;
        }

        return { obs: this.getNormalizedState(), reward, done };
    }
}