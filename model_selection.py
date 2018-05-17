import statsmodels.formula.api as smf

def create_formula(predictors, target):
    formula = target + '~'
    for x in predictors:
        formula = formula + x + '+'

    formula = formula[:-1]
    
    return formula

def criteria_based_backward_selection(predictors, target, data, criteria='rsquared_adj'):
    formula = create_formula(predictors, target)
    lr = smf.ols(formula=formula, data=data).fit()
    
    if criteria == 'rsquared_adj':
        metric = lr.rsquared_adj
    elif criteria == 'aic':
        metric = lr.aic
        
    score = metric
    step = 0
    while True:
        # initialize the element to be dropped 
        # and the maximum value of each iteration
        dropped = ''
        max_value = 0
        for x in predictors:
            # make a copy of list
            predictors_new = predictors.copy()
            predictors_new.remove(x)

            formula = create_formula(predictors_new, target)
            lr = smf.ols(formula=formula, data=data).fit()
            
            if criteria == 'rsquared_adj':
                value = lr.rsquared_adj
            elif criteria == 'aic':
                value = lr.aic
                
            # if the value is greater than current greatest value
            # update the max_value and store the variable to be dropped
            if value > max_value:
                max_value = value
                dropped = x

        if max_value < score:
            # if max_value of this iteration is smaller than score
            # quit the loop
            break
            
        step = step + 1
        # update the lower bound of score
        score = max_value
        # drop the corresponding element
        predictors.remove(dropped)
        print('Step {}:  {} is dropped ------ metirc: {}'.format(step, dropped, score))

    print(create_formula(predictors, target)) 
    
    return create_formula(predictors, target)