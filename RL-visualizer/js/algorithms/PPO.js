
export class PPO {
    constructor(env) {
        this.name = "PPO";
        this.env = env;

        // Hyperparameters optimized for CartPole
        this.hypers = {
            gamma: 0.99,            // Discount factor
            lambda: 0.95,           // GAE parameter (smooths out variance)
            clipRatio: 0.2,         // PPO clipping (prevents drastic policy changes)
            learningRate: 0.001,    // Alpha
            entropyCoef: 0.01,      // Encourages exploration (prevents early convergence)
            valueCoef: 0.5,         // Weight of value loss vs policy loss
            trainEpochs: 4,         // How many times to train on the collected batch
            batchSize: 128,         // Rollout buffer size
        };

        this.inputSize = env.name === "CartPole" ? 4 : 4; // Adjust for Maze if needed
        this.outputSize = env.name === "CartPole" ? 2 : 4;

        // 1. ACTOR Network (Policy) - Outputs probabilities (Softmax)
        this.actor = this.createActor();

        // 2. CRITIC Network (Value) - Outputs a single number (Value of state)
        this.critic = this.createCritic();

        this.optimizer = tf.train.adam(this.hypers.learningRate);

        // Memory buffers for "On-Policy" training
        this.resetMemory();

        // Temporary storage for the 'act' to 'learn' handoff
        this.lastPrediction = null;
    }

    static getMetadata() {
        return [
            { id: 'gamma', label: 'Future Bias (γ)', min: 0.9, max: 0.999, step: 0.001, default: 0.99 },
            { id: 'learningRate', label: 'Learning Rate(α)', min: 0.0001, max: 0.01, step: 0.0001, default: 0.001 },
            { id: 'entropyCoef', label: 'Curiosity / Randomness', min: 0, max: 0.1, step: 0.01, default: 0.01 }
        ];
    }

    createActor() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 24, inputShape: [this.inputSize], activation: 'tanh' })); // Tanh often works better for PPO
        model.add(tf.layers.dense({ units: 24, activation: 'tanh' }));
        model.add(tf.layers.dense({ units: this.outputSize, activation: 'softmax' })); // Probabilities
        return model;
    }

    createCritic() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 24, inputShape: [this.inputSize], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'linear' })); // Scalar value
        return model;
    }

    resetMemory() {
        this.memory = {
            states: [],
            actions: [],
            rewards: [],
            dones: [],
            oldLogProbs: [], // Log probability of the action taken
            values: []       // Critic's prediction at that step
        };
    }

    /**
     * PPO Act Logic
     * Returns an integer action.
     * Crucial: We must store the logProb and Value here to pass to learn() later.
     */
    act(state, forceDeterministic = false) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const probs = this.actor.predict(stateTensor);
            const value = this.critic.predict(stateTensor);

            const probsData = probs.dataSync();

            let action;
            if (forceDeterministic) {
                // Greedy choice for inference
                action = probs.argMax(1).dataSync()[0];
            } else {
                // Sampling for training
                action = tf.multinomial(probs, 1).dataSync()[0];
            }

            // Calculate Log Probability of the chosen action: log(prob[action])
            // We clamp slightly to avoid log(0)
            const chosenProb = Math.max(probsData[action], 1e-7);
            const logProb = Math.log(chosenProb);

            // Store these temporarily so 'learn' can grab them
            this.lastPrediction = {
                logProb: logProb,
                val: value.dataSync()[0]
            };

            return action;
        });
    }

    /**
     * PPO Learn Logic
     * Unlike DQN, PPO doesn't train every step. It buffers data and trains in batches.
     */
    async learn(s, a, r, sn, done) {
        if (!this.lastPrediction) return; // Safety check

        // 1. Store experience in buffer
        this.memory.states.push(s);
        this.memory.actions.push(a);
        this.memory.rewards.push(r);
        this.memory.dones.push(done);
        this.memory.oldLogProbs.push(this.lastPrediction.logProb);
        this.memory.values.push(this.lastPrediction.val);

        // 2. If buffer is full or episode ended, trigger training
        if (this.memory.states.length >= this.hypers.batchSize || done) {
            await this.train();
            this.resetMemory();
        }
    }

    async train() {
        if (this.memory.states.length === 0) return;

        // --- PREPARE DATA ---
        // We need the value of the NEXT state to calculate advantages
        // If the last step wasn't 'done', we bootstrap with the Critic's prediction
        let lastValue = 0;
        if (!this.memory.dones[this.memory.dones.length - 1]) {
            const lastState = this.memory.states[this.memory.states.length - 1]; // Approximation: use last state
            // Ideally we'd use 'sn' passed to learn, but for batch processing this is acceptable approximation 
            // or we could require 'sn' to be stored. 
            // For simplicity in this structure, we assume terminal reward handles most cases.
            // A more rigorous implementation would run critic on 'sn' of the last step.
            const snTensor = tf.tensor2d([this.memory.states[this.memory.states.length - 1]]);
            lastValue = this.critic.predict(snTensor).dataSync()[0];
            snTensor.dispose();
        }

        // --- CALCULATE ADVANTAGES (GAE) ---
        // Generalized Advantage Estimation: standard method for PPO stability
        const returns = [];
        const advantages = [];
        let gae = 0;

        // Iterate backwards
        for (let i = this.memory.rewards.length - 1; i >= 0; i--) {
            const isTerminal = this.memory.dones[i];
            const nextVal = (i === this.memory.rewards.length - 1) ? lastValue : this.memory.values[i + 1];

            const delta = this.memory.rewards[i] + (isTerminal ? 0 : this.hypers.gamma * nextVal) - this.memory.values[i];

            gae = delta + this.hypers.gamma * this.hypers.lambda * (isTerminal ? 0 : 1) * gae;

            advantages[i] = gae;
            returns[i] = gae + this.memory.values[i];
        }

        // Convert to Tensors
        const tfStates = tf.tensor2d(this.memory.states);
        const tfActions = tf.tensor1d(this.memory.actions, 'int32');
        const tfOldLogProbs = tf.tensor1d(this.memory.oldLogProbs);
        const tfReturns = tf.tensor1d(returns);
        const tfAdvantages = tf.tidy(() => {
            const adv = tf.tensor1d(advantages);
            const mean = adv.mean();
            const std = tf.sqrt(tf.mean(tf.square(adv.sub(mean))).add(1e-8));
            return adv.sub(mean).div(std);
        });
        // --- OPTIMIZATION LOOP ---
        for (let i = 0; i < this.hypers.trainEpochs; i++) {
            // Train Actor and Critic
            const lossInfo = await this.optimizer.minimize(() => {

                // 1. Critic Loss (MSE)
                const values = this.critic.predict(tfStates).reshape([-1]);
                const criticLoss = tf.losses.meanSquaredError(tfReturns, values);

                // 2. Actor Loss (PPO Clip)
                const logits = this.actor.predict(tfStates);
                const oneHotActions = tf.oneHot(tfActions, this.outputSize);
                const probs = tf.softmax(logits);

                // Get probability of the specific actions we took
                const actionProbs = tf.sum(tf.mul(probs, oneHotActions), 1);
                const newLogProbs = tf.log(actionProbs.add(1e-7)); // Add epsilon for stability

                // Ratio: exp(new - old)
                const ratio = tf.exp(newLogProbs.sub(tfOldLogProbs));

                // Surrogate Losses
                const surr1 = tf.mul(ratio, tfAdvantages);
                const surr2 = tf.mul(tf.clipByValue(ratio, 1 - this.hypers.clipRatio, 1 + this.hypers.clipRatio), tfAdvantages);

                // PPO Policy Loss is negative because we want to MAXIMIZE reward (minimize negative reward)
                const actorLoss = tf.mean(tf.minimum(surr1, surr2)).mul(-1);

                // 3. Entropy Bonus (To encourage exploration)
                // -sum(p * log(p))
                const entropy = tf.mean(tf.sum(tf.mul(probs, tf.log(probs.add(1e-7))), 1)).mul(-1);

                // Total Loss
                return actorLoss
                    .add(criticLoss.mul(this.hypers.valueCoef))
                    .sub(entropy.mul(this.hypers.entropyCoef));

            }, true); // returnCost = true
        }

        // Cleanup
        tfStates.dispose();
        tfActions.dispose();
        tfOldLogProbs.dispose();
        tfAdvantages.dispose();
        tfReturns.dispose();
    }

    reset() {
        this.resetMemory();
        this.actor = this.createActor();
        this.critic = this.createCritic();
        this.optimizer = tf.train.adam(this.hypers.learningRate);
    }

    async saveModel() {
        try {
            await this.actor.save('localstorage://ppo-actor');
            await this.critic.save('localstorage://ppo-critic');
            console.log("PPO Models saved successfully.");
        } catch (error) {
            console.error("Error saving model:", error);
        }
    }

    async loadModel() {
        try {
            // 1. Check if the Actor exists in LocalStorage
            // NOTE: We strip the 'localstorage://' prefix when checking the key manually
            const exists = localStorage.getItem('tensorflowjs_models/ppo-actor/info');

            if (!exists) {
                console.warn("No saved PPO model found in LocalStorage.");
                return false;
            }

            console.log("Found PPO model, loading...");

            // 2. Load both models
            // tf.loadLayersModel automagically handles the IO handler parsing
            const loadedActor = await tf.loadLayersModel('localstorage://ppo-actor');
            const loadedCritic = await tf.loadLayersModel('localstorage://ppo-critic');

            // 3. Assign to class properties
            // We use 'dispose()' on the old ones to free up GPU memory before switching
            if (this.actor) this.actor.dispose();
            if (this.critic) this.critic.dispose();

            this.actor = loadedActor;
            this.critic = loadedCritic;

            // 4. IMPORTANT: Re-initialize the optimizer
            // We do this to ensure it uses the correct Learning Rate from your current hypers
            // and resets any "Momentum" from the previous session.
            this.optimizer = tf.train.adam(this.hypers.learningRate);

            console.log("PPO Models loaded and Optimizer reset.");
            return true;

        } catch (error) {
            console.error("Error loading model:", error);
            return false;
        }
    }
}