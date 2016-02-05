/* Functions pertinent to the outer simulation steps */

#include <math.h>
#include <stdio.h>
#include "lbm.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void timestep(const param_t params, const accel_area_t accel_area, lbm_context_t lbm_context,
    speed_t* cells, speed_t* tmp_cells, int* obstacles, int timestep)
{    
    cl_int err;
    /*
    *   TODO
    *   Run OpenCL kernels on the device
    */
        // Execute the kernel 

        const size_t global[2] = {params.global_size_y, params.global_size_x};
        const size_t local[2] = {32, 32};
        err = clEnqueueNDRangeKernel(
            lbm_context.queue,
            lbm_context.propagate,
            2, NULL,
            global, local,
            0, NULL, NULL);
       if (CL_SUCCESS != err) DIE("OpenCL error %d getting clEnqueueNDRangeKernel", err);

    err  = clSetKernelArg(lbm_context.collision, 6, sizeof(int), &timestep);
     if (CL_SUCCESS != err) DIE("OpenCL error %d getting d_timestep", err); 

        const size_t global2 = (params.global_size_y*params.global_size_x);
        //HARD CODE---CHANGE TO NUMBER OPTIMAL LOCAL SIZE
        //-----------------------------------------------
        //-----------------------------------------------
        const size_t localCollision = 256;
        //-----------------------------------------------
        //-----------------------------------------------
        err = clEnqueueNDRangeKernel(
            lbm_context.queue,
            lbm_context.collision,
            1, NULL,
            &global2, &localCollision,
            0, NULL, NULL);
       if (CL_SUCCESS != err) DIE("OpenCL error %d getting clEnqueueNDRangeKernel", err);

}


float av_velocity(const param_t params, speed_t* cells, int* obstacles)
{
    int    ii,jj,kk;       /* generic counters */
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u;          /* accumulated magnitudes of velocity for each cell */

    float local_density;  /* total density in cell */
    float u_x;            /* x-component of velocity for current cell */
    float u_y;            /* y-component of velocity for current cell */

    /* initialise */
    tot_u = 0.0;

    /* loop over all non-blocked cells */
    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.nx; jj++)
        {
            /* ignore occupied cells */
            if (!obstacles[ii*params.nx + jj])
            {
                /* local density total */
                local_density = 0.0;

                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[ii*params.nx + jj].speeds[kk];
                }

                /* x-component of velocity */
                u_x = (cells[ii*params.nx + jj].speeds[1] +
                        cells[ii*params.nx + jj].speeds[5] +
                        cells[ii*params.nx + jj].speeds[8]
                    - (cells[ii*params.nx + jj].speeds[3] +
                        cells[ii*params.nx + jj].speeds[6] +
                        cells[ii*params.nx + jj].speeds[7])) /
                    local_density;

                /* compute y velocity component */
                u_y = (cells[ii*params.nx + jj].speeds[2] +
                        cells[ii*params.nx + jj].speeds[5] +
                        cells[ii*params.nx + jj].speeds[6]
                    - (cells[ii*params.nx + jj].speeds[4] +
                        cells[ii*params.nx + jj].speeds[7] +
                        cells[ii*params.nx + jj].speeds[8])) /
                    local_density;

                /* accumulate the norm of x- and y- velocity components */
                tot_u += sqrt(u_x*u_x + u_y*u_y);
                /* increase counter of inspected cells */
                ++tot_cells;
            }
        }
    }

    return tot_u / (float)tot_cells;
}