export class DQN {
    constructor(env) {
        this.name = "DQL";
        this.hypers = {
            epsilon: 1.0,
            alpha: 0.001,
            gamma: 0.95,
            epsilonDecay: 0.995, 
            batchSize: 64,
            targetUpdateFreq: 500 
        };
        this.memory = [];
        this.isTraining = false;

        this.model = this.createModel(env);

        this.targetModel = this.createModel(env);
        this.updateTargetModel();

        this.envReference = env;
        this.learnStepCount = 0; 
    }

    static getMetadata() {
        return [
            { id: 'epsilon', label: 'Curiosity (ε)', min: 0, max: 1, step: 0.01, default: 1.0 },
            { id: 'alpha', label: 'Learning Rate (α)', min: 0.0001, max: 0.01, step: 0.0001, default: 0.001 },
            { id: 'gamma', label: 'Future Bias (γ)', min: 0, max: 1, step: 0.01, default: 0.95 }
        ];
    }
    async saveModel() {
        try {

            await this.model.save('localstorage://cartpole-dqn');

            console.log("Model saved successfully.");
        } catch (error) {
            console.error("Error saving model:", error);
        }
    }
    async loadModel() {
        try {
            const modelName = 'cartpole-dqn';
            const modelPath = `localstorage://${modelName}`;

            // 1. Check if the model exists in LocalStorage
            // TensorFlow.js adds 'tensorflowjs_models/' prefix to the keys
            const exists = localStorage.getItem(`tensorflowjs_models/${modelName}/info`);

            if (!exists) {
                console.warn(`No saved model found at ${modelPath}. Skipping load.`);
                return false; // Return false so your UI knows nothing was loaded
            }

            // 2. Load the model
            const loadedModel = await tf.loadLayersModel(modelPath);

            // 3. Re-compile it
            loadedModel.compile({
                optimizer: tf.train.adam(this.hypers.alpha),
                loss: tf.losses.huberLoss
            });

            // 4. Assign to networks
            this.model = loadedModel;
            this.updateTargetModel();

            console.log("Model loaded successfully from LocalStorage.");
            return true;
        } catch (error) {
            console.error("Error loading model:", error);
            return false;
        }
    }
    createModel(env) {
        const inputSize = env.name === "CartPole" ? 4 : 4;
        const outputSize = env.name === "CartPole" ? 2 : 4;

        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 24, inputShape: [inputSize], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
        model.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));

        model.compile({
            optimizer: tf.train.adam(this.hypers.alpha),
            loss: tf.losses.huberLoss
        });
        this.outputSize = outputSize;
        return model;
    }

    updateTargetModel() {
        // tf.tidy not needed here as getWeights returns distinct tensors we need to set
        const weights = this.model.getWeights();
        this.targetModel.setWeights(weights);
        console.log("Target Network Updated");
    }

    // ... normalizeState() remains the same ...
    normalizeState(state) {
        if (this.envReference.name === "CartPole") return state;
        return [
            state[0] / this.envReference.cols,
            state[1] / this.envReference.rows,
            state[2] / this.envReference.cols,
            state[3] / this.envReference.rows
        ];
    }

    act(state, forceDeterministic = false) {
        if (!forceDeterministic && Math.random() < this.hypers.epsilon) {
            return Math.floor(Math.random() * this.outputSize);
        }
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state]);
            const prediction = this.model.predict(stateTensor);
            return prediction.argMax(1).dataSync()[0];
        });
    }

    async learn(s, a, r, sn, done) {
        this.memory.push({ s, a, r, sn, done });
        if (this.memory.length > 2000) this.memory.shift();

        // 3. AUTO-UPDATE TARGET NETWORK
        this.learnStepCount++;
        if (this.learnStepCount % this.hypers.targetUpdateFreq === 0) {
            this.updateTargetModel();
        }

        if (this.memory.length > this.hypers.batchSize && !this.isTraining) {
            this.isTraining = true;
            try { await this.replay(); }
            finally { this.isTraining = false; }
        }

        if (this.hypers.epsilon > 0.01) {
            this.hypers.epsilon *= this.hypers.epsilonDecay;
        }
    }

    async replay() {
        // Ensure we have enough memory
        if (this.memory.length < this.hypers.batchSize) return;

        const batch = _.sampleSize(this.memory, this.hypers.batchSize);

        const { statesTensor, targetsTensor } = tf.tidy(() => {
            const states = batch.map(m => m.s);
            const nextStates = batch.map(m => m.sn);
            const qValues = this.model.predict(tf.tensor2d(states)).arraySync();

            const nextQ_Main = this.model.predict(tf.tensor2d(nextStates)).arraySync();

            const nextQ_Target = this.targetModel.predict(tf.tensor2d(nextStates)).arraySync();


            const x = [];
            const y = [];

            for (let i = 0; i < batch.length; i++) {
                const { a, r, done } = batch[i];
                let target = r;

                if (!done) {

                    const bestNextActionIdx = nextQ_Main[i].indexOf(Math.max(...nextQ_Main[i]));

                    const valueOfBestAction = nextQ_Target[i][bestNextActionIdx];

                    target = r + this.hypers.gamma * valueOfBestAction;
                }

                const currentQ = [...qValues[i]];
                currentQ[a] = target;

                x.push(batch[i].s);
                y.push(currentQ);
            }

            return {
                statesTensor: tf.tensor2d(x),
                targetsTensor: tf.tensor2d(y)
            };
        });

        // Fit the model
        await this.model.fit(statesTensor, targetsTensor, {
            epochs: 1,
            verbose: 0
        });

        // Cleanup
        statesTensor.dispose();
        targetsTensor.dispose();
    }
    reset() {
        this.memory = [];
        this.hypers.epsilon = 1.0;
        this.isTraining = false;
        this.learnStepCount = 0;
        this.model = this.createModel(this.envReference);
        this.updateTargetModel(); 
    }
}