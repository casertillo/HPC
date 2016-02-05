#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

typedef struct {
    float speeds[NSPEEDS];
} speed_t;

typedef struct {
    unsigned int nx;            
    unsigned int ny;            
    unsigned int max_iters;      
    unsigned int reynolds_dim;  
    float density;       
    float accel;        
    float omega;
    unsigned int tot_cells; 
    unsigned int num_groups;     
    unsigned int global_size_x;
    unsigned int global_size_y; 
} param_t;

typedef enum { ACCEL_ROW=0, ACCEL_COLUMN=1 } accel_e;
typedef struct {
    unsigned int col_or_row;
    unsigned int idx;
} accel_area_t;

void reduce(                                          
   __local  float*,                          
   __global float*, 
   int timestep, int num_groups);

__kernel void propagate(const param_t params, const accel_area_t accel_area, __global speed_t* cells, __global speed_t* tmp_cells, __global int* obstacles)
{
    unsigned int ii, jj, inc = 0, aa = accel_area.idx, y_n,y_s, x_e,
    x_w, pos, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8;
    const float w1 = params.density * params.accel / 9.0;
    const float w2 = params.density * params.accel / 36.0;
    
    //define where is the acceleration
    inc = accel_area.col_or_row != ACCEL_COLUMN ? 1 : 0;

    //if to check global size and avoid to do more work on large pipe
    if(get_global_id(0) < 1000)
    //---------------------------------------------------------------
    {
        //if to check global size and avoid to do more work on large pipe
        if(get_global_id(1)<1000)
        //---------------------------------------------------------------
        {
        const unsigned int ii = get_global_id(0);
        y_n = (ii + 1) % params.ny;
        y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
        
        const unsigned int jj = get_global_id(1);
            pos = ii*params.nx + jj;
            speed_t buff_cell = cells[pos];
            x_e = (jj + 1) % params.nx;
            x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);

            pos1 = ii *params.nx + x_e;
            pos2 = y_n*params.nx + jj;
            pos3 = ii *params.nx + x_w;
            pos4 = y_s*params.nx + jj;
            pos5 = y_n*params.nx + x_e;
            pos6 = y_n*params.nx + x_w;
            pos7 = y_s*params.nx + x_w;
            pos8 = y_s*params.nx + x_e;
            if(aa == ii)
            {
                if(inc == 1)
                {
                    if(!obstacles[pos])
                    {
                        buff_cell.speeds[1] += w1;
                        buff_cell.speeds[5] += w2;
                        buff_cell.speeds[8] += w2;
                        
                        buff_cell.speeds[3] -= w1;
                        buff_cell.speeds[6] -= w2;
                        buff_cell.speeds[7] -= w2;
                    }
                }
            }
            if(aa == jj && inc == 0)
            {
                if(inc == 0)
                {
                    if(!obstacles[pos])
                    {
                        buff_cell.speeds[2] += w1;
                        buff_cell.speeds[5] += w2;
                        buff_cell.speeds[6] += w2;

                        buff_cell.speeds[4] -= w1;
                        buff_cell.speeds[7] -= w2;
                        buff_cell.speeds[8] -= w2;
                    }
                }
            }
            tmp_cells[pos].speeds[0]  = buff_cell.speeds[0]; 
            tmp_cells[pos1].speeds[1] = buff_cell.speeds[1]; 
            tmp_cells[pos2].speeds[2]  = buff_cell.speeds[2]; 
            tmp_cells[pos3].speeds[3] = buff_cell.speeds[3]; 
            tmp_cells[pos4].speeds[4]  = buff_cell.speeds[4]; 
            tmp_cells[pos5].speeds[5] = buff_cell.speeds[5]; 
            tmp_cells[pos6].speeds[6] = buff_cell.speeds[6]; 
            tmp_cells[pos7].speeds[7] = buff_cell.speeds[7]; 
            tmp_cells[pos8].speeds[8] = buff_cell.speeds[8];
        }
    }
}


__kernel void collision(
    const param_t params, 
    __global speed_t* cells, 
    __global speed_t* restrict tmp_cells, 
    __global int* restrict obstacles,
    __global float* av_vels,
    __local  float* local_sums,
    int timestep)
{
    float c_sq = 1.0/3.0, w0 = 4.0/9.0, w1 = 1.0/9.0, w2 = 1.0/36.0, c_2sq = 2.0 * c_sq,
           c_22sq = 2.0 * c_sq * c_sq, u_x,u_y, u_sq, local_density; 
                                                    
   //IF only used to large pipe-------------------------------------                                          
   if(get_global_id(0) < 1000000)
   //--------------------------------------------------------------- 
   {
   const int ii = get_global_id(0);
   //jj = get_global_id(1)
   speed_t buff_tmp = tmp_cells[ii];
   speed_t buff_cell = cells[ii];
   unsigned int obs = obstacles[ii];
   unsigned int local_index = get_local_id(0);
   float tot_u = 0.0;
        if (!obs)
        {
           /* compute local density total */
            local_density = 0.0;
            local_density = buff_tmp.speeds[0]+buff_tmp.speeds[1]+buff_tmp.speeds[2]+buff_tmp.speeds[3]+buff_tmp.speeds[4]
                         +buff_tmp.speeds[5]+buff_tmp.speeds[6]+buff_tmp.speeds[7]+buff_tmp.speeds[8];

            /* compute x velocity component */
            u_x = (buff_tmp.speeds[1] +buff_tmp.speeds[5] +buff_tmp.speeds[8]
               - (buff_tmp.speeds[3] +buff_tmp.speeds[6] +buff_tmp.speeds[7]))/ local_density;

           /* compute y velocity component */
            u_y = (buff_tmp.speeds[2] +buff_tmp.speeds[5] +buff_tmp.speeds[6]
               - (buff_tmp.speeds[4] +buff_tmp.speeds[7] +buff_tmp.speeds[8]))/ local_density;

            /* velocity squared */
            u_sq = u_x*u_x+ u_y*u_y;
            /* directional velocity components */
            /* accumulate the norm of x- and y- velocity components */
            tot_u += sqrt(u_sq);
            
            /* relaxation step */
            buff_cell.speeds[0] = buff_tmp.speeds[0] + params.omega * ((w0 * local_density * (1.0 - u_sq / c_2sq)) - buff_tmp.speeds[0]);
            buff_cell.speeds[1] = buff_tmp.speeds[1] + params.omega * ((w1 * local_density * (1.0 + u_x / c_sq+ (u_x*u_x) / c_22sq- u_sq / c_2sq)) - buff_tmp.speeds[1]);
            buff_cell.speeds[2] = buff_tmp.speeds[2] + params.omega * ((w1 * local_density * (1.0 + u_y / c_sq+ (u_y*u_y) / c_22sq- u_sq / c_2sq)) - buff_tmp.speeds[2]);
            buff_cell.speeds[3] = buff_tmp.speeds[3] + params.omega * ((w1 * local_density * (1.0 - u_x / c_sq+ (u_x*u_x) / c_22sq- u_sq / c_2sq)) - buff_tmp.speeds[3]);
            buff_cell.speeds[4] = buff_tmp.speeds[4] + params.omega * ((w1 * local_density * (1.0 - u_y / c_sq+ (u_y*u_y) / c_22sq- u_sq / c_2sq)) - buff_tmp.speeds[4]);
            buff_cell.speeds[5] = buff_tmp.speeds[5] + params.omega * ((w2 * local_density * (1.0 + ( u_x + u_y) / c_sq+ ((u_x + u_y)*(u_x + u_y)) / c_22sq- u_sq / c_2sq)) - buff_tmp.speeds[5]);
            buff_cell.speeds[6] = buff_tmp.speeds[6] + params.omega * ((w2 * local_density * (1.0 + ( u_y - u_x) / c_sq+ ((u_y - u_x)*(u_y - u_x)) / c_22sq- u_sq / c_2sq)) - buff_tmp.speeds[6]);
            buff_cell.speeds[7] = buff_tmp.speeds[7] + params.omega * ((w2 * local_density * (1.0 + (-u_x - u_y) / c_sq+ ((-u_x - u_y)*(-u_x - u_y)) / c_22sq- u_sq / c_2sq)) - buff_tmp.speeds[7]);
            buff_cell.speeds[8] = buff_tmp.speeds[8] + params.omega * ((w2 * local_density * (1.0 + ( u_x - u_y) / c_sq+ ((u_x - u_y)*(u_x - u_y)) / c_22sq- u_sq / c_2sq)) - buff_tmp.speeds[8]);
            
        }
        else
        {
            buff_cell.speeds[1] = buff_tmp.speeds[3];
            buff_cell.speeds[2] = buff_tmp.speeds[4];
            buff_cell.speeds[3] = buff_tmp.speeds[1];
            buff_cell.speeds[4] = buff_tmp.speeds[2];
            buff_cell.speeds[5] = buff_tmp.speeds[7];
            buff_cell.speeds[6] = buff_tmp.speeds[8];
            buff_cell.speeds[7] = buff_tmp.speeds[5];
            buff_cell.speeds[8] = buff_tmp.speeds[6];
        }
        cells[ii] = buff_cell;
        local_sums[local_index] = tot_u;
        barrier(CLK_LOCAL_MEM_FENCE);
        reduce(local_sums, av_vels, timestep, params.num_groups); 
    }
        

 
   //----------------------------------------------
}
void reduce(__local float* local_sums, __global float* av_vels, int timestep, int num_groups)
{
   float sum;                              
   unsigned int i; 

   unsigned int num_wrk_items  = get_local_size(0);                 
   unsigned int local_id       = get_local_id(0); 
   unsigned int group_id       = get_group_id(0);                   
                           
   if (local_id == 0) {                      
      sum = 0.0;
      for (i=0; i<num_wrk_items; i++) {        
          sum += local_sums[i];                                                     
        }
        
        av_vels[group_id+(timestep*num_groups)] = sum;
    }   
}