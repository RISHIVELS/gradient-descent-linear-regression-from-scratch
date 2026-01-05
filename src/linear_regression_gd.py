import math

class LinearRegressionGradientDescent:
    # Initialised the parameters
    def __init__ (self):
        self.isFitted = False
        self.beta0 = 0
        self.beta1 = 0
        self.beta0_history = []
        self.beta1_history = []
        self.slope_history = []          # track beta1 updates
        self.loss_history = []

    def __str__(self):
        return f"Beta0 : {self.beta0}, Beta1 : {self.beta1} {self.beta0_history} hello"

    # compute the beta0 and beta1 values
    def fit (self,x,y, alpha=0.001, epochs=1000, tol=1e-6):
        
        n = len(x)

        # reset histories for each fit
        self.beta0_history = []
        self.beta1_history = []
        self.slope_history = []
        self.loss_history = []

        beta0 = 0.0
        beta1 = 0.0

        for _ in range(epochs):
            y_pred = beta0 + beta1 * x
            error = y_pred - y
            # gradients
            beta0_der = (1/n) * error.sum()
            beta1_der = (1/n) * (error * x).sum()
            # gradient step
            beta0_new = beta0 - alpha * beta0_der
            beta1_new = beta1 - alpha * beta1_der

            # record
            self.beta0_history.append(beta0_new)
            self.beta1_history.append(beta1_new)
            self.slope_history.append(beta1_der)
            loss = (1/(2*n)) * ((error) ** 2).sum() # Mean Squared Error
            self.loss_history.append(loss)

            # check convergence
            if abs(beta0_new - beta0) < tol and abs(beta1_new - beta1) < tol:
                beta0, beta1 = beta0_new, beta1_new
                break

            beta0, beta1 = beta0_new, beta1_new

        self.isFitted = True
        self.beta0 = beta0
        self.beta1 = beta1
        

    def predict(self,x):
        # If not trained throw error
        if (not self.isFitted):
            raise Exception('Model is not Fitted. Please train the model first!!!')
        self.y_pred = self.beta0 + self.beta1*x;
        return self.y_pred

    # Metrics
    # Mean square Error
    def mean_square_error(self,y_true,y_pred):
        if (len(y_true) != len(y_pred)):
            raise Exception("The size of y_pred and y_true does not match!!!")
        self.n = len(y_true)
        self.mse = 1/self.n * sum((y_true - y_pred)**2)
        return self.mse
        
    # Mean absolute error
    def mean_absolute_error(self,y_true,y_pred):
        if (len(y_true) != len(y_pred)):
            raise Exception("The size of y_pred and y_true does not match!!!")
        self.n = len(y_true)
        self.mae = 1/self.n * sum(abs(y_true - y_pred))
        return self.mae

    # Root mean square error
    def rootmean_square_error(self,y_true,y_pred):
        self.y_true = y_true
        if (len(y_true) != len(y_pred)):
            raise Exception("The size of y_pred and y_true does not match!!!")
        self.n = len(y_true)
        self.rmse = math.sqrt(sum((y_true - y_pred)**2)/self.n)
        return self.rmse

    def r2_score (self,y_true,y_pred):
        if (len(y_true) != len(y_pred)):
            raise Exception("The size of y_pred and y_true does not match!!!")
        self.numerator = sum((y_true - y_pred)**2)
        self.denomerator = sum((y_true - y_true.mean())**2)
        self.r2 = 1 - (self.numerator/self.denomerator)
        return self.r2

    # summart metric of the model
    def summary(self,y,y_pred):
        print("RMSE : ",self.rootmean_square_error(y,y_pred))
        print("R2_SCORE : ",self.r2_score(y,y_pred))
        print("MAE : ",self.mean_absolute_error(y,y_pred))
        print("MSE : ",self.mean_square_error(y,y_pred))
            