#include <math.h>

#include "admm.h"

extern Parameters params;
extern FILE *output;
extern FILE *traindata;
extern int BabPbSize;

extern double root_basicSDP_bound;
extern double root_bound;
extern int maxcut_size;
extern double maxcut_density;
extern SVM * svm;

extern double TIME;                 

extern Triangle_Inequality *Cuts;               // vector of triangle inequality constraints
extern Pentagonal_Inequality *Pent_Cuts;        // vector of pentagonal inequality constraints
extern Heptagonal_Inequality *Hepta_Cuts;       // vector of heptagonal inequality constraints

extern double *X;                   // current X
extern double *y;                   // dual multiplier for diagonal constraints
extern double *Q;                   // Z matrix in admm method
extern double *s;                   // nonnegative dual multiplier to constraint u >= 0
                                    // used for purging of cuts
extern double f;                    // function value of relaxation

extern double diff;		            // difference between basic SDP relaxation and bound with added cutting planes

/******** main bounding routine calling ADMM method ********/
double SDPbound(BabNode *node, Problem *SP, Problem *PP, int rank) {
    double sdpstartime = MPI_Wtime();
    int index;                      // helps to store the fractional solution in the node
    double bound;                   // f + fixedvalue
    double gap;                     // difference between best lower bound and upper bound
    double oldf;                    // stores f from previous iteration 
    int x[BabPbSize];               // vector for heuristic
    double viol3 = 0.0;             // maximum violation of triangle inequalities
    double viol5 = 0.0;             // maximum violation of pentagonal inequalities
    double viol7 = 0.0;             // maximum violation of heptagonal inequalities
    int count = 0;                  // number of iterations (adding and purging of cutting planes)
    int nbit;                       // number of iterations in ADMM method

    int triag;                      // starting index for pentagonal inequalities in vectors t and s 
    int penta;                      // starting index for heptagonal inequalities in vectors t and s

    /* stopping conditions */
    int done = 0;                   
    int giveup = 0;                                   
    int prune = 0;

    // Parameters
    double sigma = params.sigma0;
    double tol = params.tol0;

    // fixed value contributes to the objective value
    double fixedvalue = getFixedValue(node, SP);

    /*** start with no cuts ***/
    // triangle inequalities
    PP->NIneq = 0; 
    int Tri_NumAdded = 0;
    int Tri_NumSubtracted = 0;

    // pentagonal inequalities
    PP->NPentIneq = 0;
    int Pent_NumAdded = 0;
    int Pent_NumSubtracted = 0;

    // heptagonal inequalities
    PP->NHeptaIneq = 0;
    int Hepta_NumAdded = 0;
    int Hepta_NumSubtracted = 0;         

    /* solve basic SDP relaxation with interior-point method */
    ipm_mc_pk(PP->L, PP->n, X, y, Q, &f, 0);


    //get the number of nonzeros in L
    int nzero = 0;
    for (int i = 0; i < PP->n; ++i) {
        for (int j = 0; j < PP->n; ++j) {
            if (fabs(PP->L[j + i * PP->n]) > 0.00001)
                nzero ++;
        }
    }


    // store basic SDP bound to compute diff in the root node
    double basic_bound = f + fixedvalue;

    // Store the fractional solution in the node    
    index = 0;
    for (int i = 0; i < BabPbSize; ++i) {
        if (node->xfixed[i]) {
            node->fracsol[i] = (double) node->sol.X[i];
        }
        else {
            // convert x (last column X) from {-1,1} to {0,1}
            node->fracsol[i] = 0.5*(X[(PP->n - 1) + index*PP->n] + 1.0); 
            ++index;
        }
    }

    /* run heuristic */
    for (int i = 0; i < BabPbSize; ++i) {
        if (node->xfixed[i]) {
            x[i] = node->sol.X[i];
        }
        else {
            x[i] = 0;
        }
    }

    runHeuristic(SP, PP, node, x);
    updateSolution(x);

    // upper bound
    bound = f + fixedvalue;

    // check pruning condition
    if ( bound < Bab_LBGet() + 1.0 ) {
        prune = 1;
        goto END;
    }

    // check if cutting planes need to be added     
    if (params.use_diff && (rank != 0) && (bound > Bab_LBGet() + diff + 1.0)) {
        giveup = 1;
        goto END;
    }

    /* separate first triangle inequality */
    viol3 = updateTriangleInequalities(PP, s, &Tri_NumAdded, &Tri_NumSubtracted);

   
    /*** Main loop ***/
    while (!done) {

        // Update iteration counter
        ++count;
        oldf = f;

        // Call ADMM solver
        ADMM_solver(PP, X, y, Q, &sigma, tol, &nbit, 0);

        // increase precision
        tol *= params.scaleTol;

        if (tol < params.minTol)
            tol = params.minTol;

        // upper bound
        bound = f + fixedvalue;

        // prune test
        prune = ( bound < Bab_LBGet() + 1.0 ) ? 1 : 0;
 
        /******** heuristic ********/
        if (!prune) {

            for (int i = 0; i < BabPbSize; ++i) {
                if (node->xfixed[i]) {
                    x[i] = node->sol.X[i];
                }
                else {
                    x[i] = 0;
                }
            }

            runHeuristic(SP, PP, node, x);
            updateSolution(x);

            prune = ( bound < Bab_LBGet() + 1.0 ) ? 1 : 0;
        }
        /***************************/

        // compute gap
        gap = bound - Bab_LBGet();

        /* check if we will not be able to prune the node */
        if (count == params.triag_iter + params.pent_iter + params.hept_iter) {
            if ( (gap - 1.0 > (oldf - f)*(params.max_outer_iter - count)))
                giveup = 1;
        }

        /* check if extra iterations can close the gap */
        if (count == params.max_outer_iter) {
            if ( gap - 1.0 > (oldf - f)*params.extra_iter )
                giveup = 1;
        }
        
        /* max number of iterations reached */
        if (count == params.max_outer_iter + params.extra_iter)
            giveup = 1; 


        /* increase number of pentagonal and heptagonal inequalities (during separation)
         * if the gap is still too big 
         */
        if ((rank != 0) && giveup && !prune) {
            params.Pent_Trials += 60;    // add 3 types * 60 = 180 pentagonal inequalities
            params.Hepta_Trials += 50;  // add 4 types * 50 = 200 heptagonal inequalities
        }

        // purge inactive cutting planes, add new inequalities
        if (!prune && !giveup) {
            
            triag = PP->NIneq;          // save number of triangle and pentagonal inequalities before purging
            penta = PP->NPentIneq;      // --> to know with which index in dual vector gamma, pentagonal
                                        // and heptagonal inequalities start!

            viol3 = updateTriangleInequalities(PP, s, &Tri_NumAdded, &Tri_NumSubtracted);
                      
            /* include pentagonal and heptagonal inequalities */          
            if ( params.include_Pent && (count >= params.triag_iter || viol3 < 0.2) )
                viol5 = updatePentagonalInequalities(PP, s, &Pent_NumAdded, &Pent_NumSubtracted, triag);  

            if ( params.include_Hepta && ( (count >= params.triag_iter + params.pent_iter) || (viol3 < 0.2 && (1 - viol5 < 0.4)) ) )
                viol7 = updateHeptagonalInequalities(PP, s, &Hepta_NumAdded, &Hepta_NumSubtracted, triag + penta);      
        }
        else {               
            Tri_NumAdded = 0;
            Tri_NumSubtracted = 0;
            Pent_NumAdded = 0;
            Pent_NumSubtracted = 0;
            Hepta_NumAdded = 0;
            Hepta_NumSubtracted = 0;
        }

        


        // Test stopping conditions
        done = 
            prune ||                         // can prune the B&B tree 
            giveup ||                        // upper bound to far away from lower bound
            (nbit >= params.ADMM_itermax);   // ADMM reached max iter

        // Store the fractional solution in the node    
        index = 0;
        for (int i = 0; i < BabPbSize; ++i) {
            if (node->xfixed[i]) {
                node->fracsol[i] = (double) node->sol.X[i];
            }
            else {
                // convert x (last column X) from {-1,1} to {0,1}
                node->fracsol[i] = 0.5*(X[(PP->n - 1) + index*PP->n] + 1.0); 
                ++index;
            }
        }

    } // end while loop

    bound = f + fixedvalue;

    // compute difference between basic SDP relaxation and bound with added cutting planes
    if (rank == 0) {
        diff = basic_bound - bound;
    }

    END:
    // if (rank ==0){
    //     fprintf(traindata, "%d %d %.3f %.3f %.3f %.3f \n", maxcut_size, PP->n, maxcut_density, bound,
    //             root_basicSDP_bound,  Bab_LBGet());
    // }
    printf("prediction fathom %d, true fathom %d ncuts %d node solution time %.3f\n", 1-giveup, prune, params.Hepta_Trials /50, MPI_Wtime() - sdpstartime);
    return bound;

}


double SDPdatacollection(BabNode *node, Problem *SP, Problem *PP, int rank) {
    int cur_Pent_Trials;
    int cur_Hepta_Trials;
    cur_Pent_Trials = params.Pent_Trials;
    cur_Hepta_Trials = params.Hepta_Trials;
    printf("calling sdp datacollection\n");
    for (int outer_iter = 0; outer_iter <= 5; outer_iter++) {
        printf("number of outer iterations %d\n", outer_iter);
        params.Pent_Trials = 60 * outer_iter;
        params.Hepta_Trials = 50 * outer_iter;
        double sdpstartime = MPI_Wtime();
        int index;                      // helps to store the fractional solution in the node
        double bound;                   // f + fixedvalue
        double gap;                     // difference between best lower bound and upper bound
        double oldf;                    // stores f from previous iteration
        int x[BabPbSize];               // vector for heuristic
        double viol3 = 0.0;             // maximum violation of triangle inequalities
        double viol5 = 0.0;             // maximum violation of pentagonal inequalities
        double viol7 = 0.0;             // maximum violation of heptagonal inequalities
        int count = 0;                  // number of iterations (adding and purging of cutting planes)
        int nbit;                       // number of iterations in ADMM method

        int triag;                      // starting index for pentagonal inequalities in vectors t and s
        int penta;                      // starting index for heptagonal inequalities in vectors t and s

        /* stopping conditions */
        int done = 0;
        int giveup = 0;
        int prune = 0;

        // Parameters
        double sigma = params.sigma0;
        double tol = params.tol0;

        // fixed value contributes to the objective value
        double fixedvalue = getFixedValue(node, SP);
        /*** start with no cuts ***/
        // triangle inequalities
        PP->NIneq = 0;
        int Tri_NumAdded = 0;
        int Tri_NumSubtracted = 0;

        // pentagonal inequalities
        PP->NPentIneq = 0;
        int Pent_NumAdded = 0;
        int Pent_NumSubtracted = 0;

        // heptagonal inequalities
        PP->NHeptaIneq = 0;
        int Hepta_NumAdded = 0;
        int Hepta_NumSubtracted = 0;

        /* solve basic SDP relaxation with interior-point method */
        ipm_mc_pk(PP->L, PP->n, X, y, Q, &f, 0);

        //get the number of nonzeros in L
        int nzero = 0;
        for (int i = 0; i < PP->n; ++i) {
            for (int j = 0; j < PP->n; ++j) {
                if (fabs(PP->L[j + i * PP->n]) > 0.00001)
                    nzero++;
            }
        }


        // store basic SDP bound to compute diff in the root node
        double basic_bound = f + fixedvalue;

        if (rank == 0){
            root_basicSDP_bound = basic_bound;
        }

        //save basic info of the node and the rootnode
        fprintf(traindata, "%d %.3f %.3f ", PP->n, basic_bound, Bab_LBGet());

//
//        fprintf(traindata, "%d %d %.3f %.3f %.3f %.3f %.3f ", maxcut_size, PP->n, maxcut_density, root_bound,
//                root_basicSDP_bound, basic_bound, Bab_LBGet());
        //save number of pent and hepta trials
        fprintf(traindata, "%d %d ", params.Pent_Trials, params.Hepta_Trials);

        // check pruning condition
        if (basic_bound < Bab_LBGet() + 1.0) {
            prune = 1;
            //save iteration info to 0 ADMM iteration
            fprintf(traindata, "0 %.3f %.3f\n", basic_bound, MPI_Wtime() - sdpstartime);
            printf("break from basic bound");
            break;
        }


        /* separate first triangle inequality */
        viol3 = updateTriangleInequalities(PP, s, &Tri_NumAdded, &Tri_NumSubtracted);

        //record the iteration stats
        double iter_time[params.max_outer_iter + params.extra_iter];
        double iter_bound[params.max_outer_iter + params.extra_iter];
        int iter_num_tri_cuts[params.max_outer_iter + params.extra_iter];
        int iter_num_pent_cuts[params.max_outer_iter + params.extra_iter];
        int iter_num_hepta_cuts[params.max_outer_iter + params.extra_iter];
        /*** Main loop ***/
        while (!done) {

            // Update iteration counter
            ++count;

            // Call ADMM solver
            ADMM_solver(PP, X, y, Q, &sigma, tol, &nbit, 0);

            // increase precision
            tol *= params.scaleTol;

            if (tol < params.minTol)
                tol = params.minTol;

            // upper bound
            bound = f + fixedvalue;

            // prune test
            prune = (bound < Bab_LBGet() + 1.0) ? 1 : 0;

            //Update iteration stats
            iter_time[count-1] = MPI_Wtime() - sdpstartime;
            iter_bound[count-1] = bound;
            iter_num_tri_cuts[count-1] = PP->NIneq;
            iter_num_pent_cuts[count-1] = PP->NPentIneq;
            iter_num_hepta_cuts[count-1] = PP->NHeptaIneq;

            // compute gap
            gap = bound - Bab_LBGet();

            /* check if we will not be able to prune the node */
            if (count == params.triag_iter + params.pent_iter + params.hept_iter) {
                if ((gap - 1.0 > (oldf - f) * (params.max_outer_iter - count)))
                    giveup = 1;
            }

            /* check if extra iterations can close the gap */
            if (count == params.max_outer_iter) {
                if (gap - 1.0 > (oldf - f) * params.extra_iter)
                    giveup = 1;
            }

            /* max number of iterations reached */
            if (count == params.max_outer_iter + params.extra_iter)
                giveup = 1;

            // purge inactive cutting planes, add new inequalities
            if (!prune && !giveup) {

                triag = PP->NIneq;          // save number of triangle and pentagonal inequalities before purging
                penta = PP->NPentIneq;      // --> to know with which index in dual vector gamma, pentagonal
                // and heptagonal inequalities start!

                viol3 = updateTriangleInequalities(PP, s, &Tri_NumAdded, &Tri_NumSubtracted);

                /* include pentagonal and heptagonal inequalities */
                if (params.include_Pent && (count >= params.triag_iter || viol3 < 0.2))
                    viol5 = updatePentagonalInequalities(PP, s, &Pent_NumAdded, &Pent_NumSubtracted, triag);

                if (params.include_Hepta &&
                    ((count >= params.triag_iter + params.pent_iter) || (viol3 < 0.2 && (1 - viol5 < 0.4))))
                    viol7 = updateHeptagonalInequalities(PP, s, &Hepta_NumAdded, &Hepta_NumSubtracted, triag + penta);
            } else {
                Tri_NumAdded = 0;
                Tri_NumSubtracted = 0;
                Pent_NumAdded = 0;
                Pent_NumSubtracted = 0;
                Hepta_NumAdded = 0;
                Hepta_NumSubtracted = 0;
            }




            // Test stopping conditions
            done =
                    prune ||                         // can prune the B&B tree
                    giveup ||                        // upper bound to far away from lower bound
                    (nbit >= params.ADMM_itermax);   // ADMM reached max iter


        } // end while loop

        bound = f + fixedvalue;

        //Save data for all the iterations
        //save final iteration, bound, and time
        fprintf(traindata, "{%d %.3f %.3f}", count, bound, MPI_Wtime()-sdpstartime);
        for (int i=0; i<count; ++i){
            fprintf(traindata, "[%d %.3f %.3f %d %d %d]", i+1, iter_bound[i], iter_time[i],
                    iter_num_tri_cuts[i], iter_num_pent_cuts[i], iter_num_hepta_cuts[i]);
        }
	fprintf(traindata, "\n");
    }
    //recover the MADAM parameter
    params.Pent_Trials = cur_Pent_Trials;
    params.Hepta_Trials = cur_Hepta_Trials;
    return 0;

}

double svm_predict(SVM * svm, double * new_feature){
    double x[6];
   for (int i =0; i<svm->n_features; i++){
      x[i] = (new_feature[i]- svm->scaler_mean[i]) / svm->scaler_std[i];
   }
   double val = svm->intercept;
   double gamma = svm->gamma;
   for (int i = 0; i < svm->n_support; i++){
      double dot_product = 0.0;
      for (int j=0; j< svm->n_features; j++){
      dot_product += (x[j] - svm->support_vectors[i][j]) * (x[j] - svm->support_vectors[i][j]);
   }
   val += svm->dual_coef[i] * exp(-gamma * dot_product);
   }
   return val;
}

/******** main bounding routine calling ADMM method ********/
double SDPboundsvm(BabNode *node, Problem *SP, Problem *PP, int rank) {
    double sdpstartime = MPI_Wtime();
    int index;                      // helps to store the fractional solution in the node
    double bound;                   // f + fixedvalue
    double gap;                     // difference between best lower bound and upper bound
    double oldf;                    // stores f from previous iteration 
    int x[BabPbSize];               // vector for heuristic
    double viol3 = 0.0;             // maximum violation of triangle inequalities
    double viol5 = 0.0;             // maximum violation of pentagonal inequalities
    double viol7 = 0.0;             // maximum violation of heptagonal inequalities
    int count = 0;                  // number of iterations (adding and purging of cutting planes)
    int nbit;                       // number of iterations in ADMM method

    int triag;                      // starting index for pentagonal inequalities in vectors t and s 
    int penta;                      // starting index for heptagonal inequalities in vectors t and s
    int ncuts = 0;

    /* stopping conditions */
    int done = 0;                   
    int giveup = 0;                                   
    int prune = 0;
    int pred_fathom = 0;

    // Parameters
    double sigma = params.sigma0;
    double tol = params.tol0;

    // fixed value contributes to the objective value
    double fixedvalue = getFixedValue(node, SP);

    /*** start with no cuts ***/
    // triangle inequalities
    PP->NIneq = 0; 
    int Tri_NumAdded = 0;
    int Tri_NumSubtracted = 0;

    // pentagonal inequalities
    PP->NPentIneq = 0;
    int Pent_NumAdded = 0;
    int Pent_NumSubtracted = 0;

    // heptagonal inequalities
    PP->NHeptaIneq = 0;
    int Hepta_NumAdded = 0;
    int Hepta_NumSubtracted = 0;         

    /* solve basic SDP relaxation with interior-point method */
    ipm_mc_pk(PP->L, PP->n, X, y, Q, &f, 0);


    //get the number of nonzeros in L
    int nzero = 0;
    for (int i = 0; i < PP->n; ++i) {
        for (int j = 0; j < PP->n; ++j) {
            if (fabs(PP->L[j + i * PP->n]) > 0.00001)
                nzero ++;
        }
    }


    // store basic SDP bound to compute diff in the root node
    double basic_bound = f + fixedvalue;
    if (rank == 0){
        root_basicSDP_bound = basic_bound;
    }

    // Store the fractional solution in the node    
    index = 0;
    for (int i = 0; i < BabPbSize; ++i) {
        if (node->xfixed[i]) {
            node->fracsol[i] = (double) node->sol.X[i];
        }
        else {
            // convert x (last column X) from {-1,1} to {0,1}
            node->fracsol[i] = 0.5*(X[(PP->n - 1) + index*PP->n] + 1.0); 
            ++index;
        }
    }

    /* run heuristic */
    for (int i = 0; i < BabPbSize; ++i) {
        if (node->xfixed[i]) {
            x[i] = node->sol.X[i];
        }
        else {
            x[i] = 0;
        }
    }

    runHeuristic(SP, PP, node, x);
    updateSolution(x);

    // upper bound
    bound = f + fixedvalue;

    // check pruning condition
    if ( bound < Bab_LBGet() + 1.0 ) {
        prune = 1;
        goto END;
    }

    // // check if cutting planes need to be added     
    // if (params.use_diff && (rank != 0) && (bound > Bab_LBGet() + diff + 1.0)) {
    //     giveup = 1;
    //     goto END;
    // }

    //thresdhold for terminating the algorithm
    double time_quantile[6] = {1.4689, 2.619, 3.4791, 4.349, 5.223, 6.147};

    if (rank !=0){
        double new_feauture[6];

        new_feauture[0] = maxcut_size - PP->n;
        new_feauture[2] = basic_bound - Bab_LBGet();
        new_feauture[3] = maxcut_density;
        new_feauture[4] = root_bound - Bab_LBGet();
        new_feauture[5] = root_basicSDP_bound - Bab_LBGet();

        pred_fathom = 0;
        for (ncuts=0; ncuts <2; ncuts++){
            new_feauture[1] = ncuts;
            double svm_start_time = MPI_Wtime();
            double pred_val = svm_predict(svm, new_feauture);
            // printf("rank %d, ncuts %d, feature val %.3f %.3f %.3f %.3f %.3f %.3f pred val %.3f inference time %.7f\n", rank, ncuts, new_feauture[0],new_feauture[1],new_feauture[2],new_feauture[3],new_feauture[4],new_feauture[5],pred_val, MPI_Wtime() - svm_start_time);
            if (pred_val > 0){
                pred_fathom = 1;
                break;
            }
        }

        if (pred_fathom){
            params.Pent_Trials = 60*ncuts;   
            params.Hepta_Trials = 50*ncuts;
        }
        else{
            giveup = 1;
            goto END;
        }
    }
    /* separate first triangle inequality */
    viol3 = updateTriangleInequalities(PP, s, &Tri_NumAdded, &Tri_NumSubtracted);

    double while_loop_start = MPI_Wtime();   
    /*** Main loop ***/
    while (!done) {

        // Update iteration counter
        ++count;
        oldf = f;

        // Call ADMM solver
        ADMM_solver(PP, X, y, Q, &sigma, tol, &nbit, 0);

        // increase precision
        tol *= params.scaleTol;

        if (tol < params.minTol)
            tol = params.minTol;

        // upper bound
        bound = f + fixedvalue;

        // prune test
        prune = ( bound < Bab_LBGet() + 1.0 ) ? 1 : 0;
 
        /******** heuristic ********/
        if (!prune) {

            for (int i = 0; i < BabPbSize; ++i) {
                if (node->xfixed[i]) {
                    x[i] = node->sol.X[i];
                }
                else {
                    x[i] = 0;
                }
            }

            runHeuristic(SP, PP, node, x);
            updateSolution(x);

            prune = ( bound < Bab_LBGet() + 1.0 ) ? 1 : 0;
        }
        /***************************/

        // compute gap
        gap = bound - Bab_LBGet();

        /*check if we are outof time limit*/
        if (ncuts >=2 && ((MPI_Wtime() - while_loop_start) > time_quantile[ncuts]|| (MPI_Wtime() - while_loop_start) / count * 9 > time_quantile[ncuts]))
            giveup = 1;
        

        /* check if we will not be able to prune the node */
        if (count == params.triag_iter + params.pent_iter + params.hept_iter) {
            if ( (gap - 1.0 > (oldf - f)*(params.max_outer_iter - count)))
                giveup = 1;
        }

        /* check if extra iterations can close the gap */
        if (count == params.max_outer_iter) {
            if ( gap - 1.0 > (oldf - f)*params.extra_iter )
                giveup = 1;
        }
        
        /* max number of iterations reached */
        if (count == params.max_outer_iter + params.extra_iter)
            giveup = 1; 


        // /* increase number of pentagonal and heptagonal inequalities (during separation)
        //  * if the gap is still too big 
        //  */
        // if ((rank != 0) && giveup && !prune) {
        //     params.Pent_Trials += 60;    // add 3 types * 60 = 180 pentagonal inequalities
        //     params.Hepta_Trials += 50;  // add 4 types * 50 = 200 heptagonal inequalities
        // }

        // purge inactive cutting planes, add new inequalities
        if (!prune && !giveup) {
            
            triag = PP->NIneq;          // save number of triangle and pentagonal inequalities before purging
            penta = PP->NPentIneq;      // --> to know with which index in dual vector gamma, pentagonal
                                        // and heptagonal inequalities start!

            viol3 = updateTriangleInequalities(PP, s, &Tri_NumAdded, &Tri_NumSubtracted);
                      
            /* include pentagonal and heptagonal inequalities */          
            if ( params.include_Pent && (count >= params.triag_iter || viol3 < 0.2) )
                viol5 = updatePentagonalInequalities(PP, s, &Pent_NumAdded, &Pent_NumSubtracted, triag);  

            if ( params.include_Hepta && ( (count >= params.triag_iter + params.pent_iter) || (viol3 < 0.2 && (1 - viol5 < 0.4)) ) )
                viol7 = updateHeptagonalInequalities(PP, s, &Hepta_NumAdded, &Hepta_NumSubtracted, triag + penta);      
        }
        else {               
            Tri_NumAdded = 0;
            Tri_NumSubtracted = 0;
            Pent_NumAdded = 0;
            Pent_NumSubtracted = 0;
            Hepta_NumAdded = 0;
            Hepta_NumSubtracted = 0;
        }

        


        // Test stopping conditions
        done = 
            prune ||                         // can prune the B&B tree 
            giveup ||                        // upper bound to far away from lower bound
            (nbit >= params.ADMM_itermax);   // ADMM reached max iter

        // Store the fractional solution in the node    
        index = 0;
        for (int i = 0; i < BabPbSize; ++i) {
            if (node->xfixed[i]) {
                node->fracsol[i] = (double) node->sol.X[i];
            }
            else {
                // convert x (last column X) from {-1,1} to {0,1}
                node->fracsol[i] = 0.5*(X[(PP->n - 1) + index*PP->n] + 1.0); 
                ++index;
            }
        }

    } // end while loop

    bound = f + fixedvalue;

    // compute difference between basic SDP relaxation and bound with added cutting planes
    if (rank == 0) {
        diff = basic_bound - bound;
    }

    END:
    printf("prediction fathom %d, true fathom %d ncuts %d node solution time %.3f\n", pred_fathom, prune, ncuts, MPI_Wtime() - sdpstartime);
    return bound;

}



/******** main bounding routine calling ADMM method ********/
double biqmacbound(BabNode *node, Problem *SP, Problem *PP, int rank) {
    double sdpstartime = MPI_Wtime();
    int index;                      // helps to store the fractional solution in the node
    double bound;                   // f + fixedvalue
    double gap;                     // difference between best lower bound and upper bound
    double oldf;                    // stores f from previous iteration 
    int x[BabPbSize];               // vector for heuristic
    double viol3 = 0.0;             // maximum violation of triangle inequalities
    int count = 0;                  // number of iterations (adding and purging of cutting planes)
    int nbit;                       // number of iterations in ADMM method

    int triag;                      // starting index for pentagonal inequalities in vectors t and s 

    /* stopping conditions */
    int done = 0;                   
    int giveup = 0;                                   
    int prune = 0;
    

    // Parameters
    double sigma = params.sigma0;
    double tol = params.tol0;

    // fixed value contributes to the objective value
    double fixedvalue = getFixedValue(node, SP);

    /*** start with no cuts ***/
    // triangle inequalities
    PP->NIneq = 0; 
    int Tri_NumAdded = 0;
    int Tri_NumSubtracted = 0;
       

    /* solve basic SDP relaxation with interior-point method */
    ipm_mc_pk(PP->L, PP->n, X, y, Q, &f, 0);


    //get the number of nonzeros in L
    int nzero = 0;
    for (int i = 0; i < PP->n; ++i) {
        for (int j = 0; j < PP->n; ++j) {
            if (fabs(PP->L[j + i * PP->n]) > 0.00001)
                nzero ++;
        }
    }


    // store basic SDP bound to compute diff in the root node
    double basic_bound = f + fixedvalue;

    // Store the fractional solution in the node    
    index = 0;
    for (int i = 0; i < BabPbSize; ++i) {
        if (node->xfixed[i]) {
            node->fracsol[i] = (double) node->sol.X[i];
        }
        else {
            // convert x (last column X) from {-1,1} to {0,1}
            node->fracsol[i] = 0.5*(X[(PP->n - 1) + index*PP->n] + 1.0); 
            ++index;
        }
    }

    /* run heuristic */
    for (int i = 0; i < BabPbSize; ++i) {
        if (node->xfixed[i]) {
            x[i] = node->sol.X[i];
        }
        else {
            x[i] = 0;
        }
    }

    runHeuristic(SP, PP, node, x);
    updateSolution(x);

    // upper bound
    bound = f + fixedvalue;

    // check pruning condition
    if ( bound < Bab_LBGet() + 1.0 ) {
        prune = 1;
        goto END;
    }

 
    /* separate first triangle inequality */
    viol3 = updateTriangleInequalities(PP, s, &Tri_NumAdded, &Tri_NumSubtracted);

    double while_loop_start = MPI_Wtime();   
    /*** Main loop ***/
    while (!done) {

        // Update iteration counter
        ++count;
        oldf = f;

        // Call ADMM solver
        ADMM_solver(PP, X, y, Q, &sigma, tol, &nbit, 0);

        // increase precision
        tol *= params.scaleTol;

        if (tol < params.minTol)
            tol = params.minTol;

        // upper bound
        bound = f + fixedvalue;

        // prune test
        prune = ( bound < Bab_LBGet() + 1.0 ) ? 1 : 0;
 
        /******** heuristic ********/
        if (!prune) {

            for (int i = 0; i < BabPbSize; ++i) {
                if (node->xfixed[i]) {
                    x[i] = node->sol.X[i];
                }
                else {
                    x[i] = 0;
                }
            }

            runHeuristic(SP, PP, node, x);
            updateSolution(x);

            prune = ( bound < Bab_LBGet() + 1.0 ) ? 1 : 0;
        }
        /***************************/

        // compute gap
        gap = bound - Bab_LBGet();


        /* check if we will not be able to prune the node */
        if (count == params.triag_iter + params.pent_iter + params.hept_iter) {
            if ( (gap - 1.0 > (oldf - f)*(params.max_outer_iter - count)))
                giveup = 1;
        }

        /* check if extra iterations can close the gap */
        if (count == params.max_outer_iter) {
            if ( gap - 1.0 > (oldf - f)*params.extra_iter )
                giveup = 1;
        }
        
        /* max number of iterations reached */
        if (count == params.max_outer_iter + params.extra_iter)
            giveup = 1; 



        // purge inactive cutting planes, add new inequalities
        if (!prune && !giveup) {
            
            triag = PP->NIneq;          // save number of triangle and pentagonal inequalities before purging

            viol3 = updateTriangleInequalities(PP, s, &Tri_NumAdded, &Tri_NumSubtracted);
                       
        }
        else {               
            Tri_NumAdded = 0;
            Tri_NumSubtracted = 0;
                   }

        


        // Test stopping conditions
        done = 
            prune ||                         // can prune the B&B tree 
            giveup ||                        // upper bound to far away from lower bound
            (nbit >= params.ADMM_itermax);   // ADMM reached max iter

        // Store the fractional solution in the node    
        index = 0;
        for (int i = 0; i < BabPbSize; ++i) {
            if (node->xfixed[i]) {
                node->fracsol[i] = (double) node->sol.X[i];
            }
            else {
                // convert x (last column X) from {-1,1} to {0,1}
                node->fracsol[i] = 0.5*(X[(PP->n - 1) + index*PP->n] + 1.0); 
                ++index;
            }
        }

    } // end while loop

    bound = f + fixedvalue;

    // compute difference between basic SDP relaxation and bound with added cutting planes

    END:

    return bound;

}