#ifndef _AMPI_BOOKKEEPING_SUPPORT_H_
#define _AMPI_BOOKKEEPING_SUPPORT_H_

#include "ampi/userIF/request.h"

/**
 * \file 
 * methods needed for internal request bookkeeping
 */ 

/**
 * \param ampiRequest is added (by deep copy) to the internal bookkeeping using the already set valueu of member plainRequest as key
 */
void BK_AMPI_put_AMPI_Request(struct AMPI_Request_S  *ampiRequest);

/**
 * \param request is used as key to look up the associated AMPI_Request_S instance which then is deep copied 
 * \param ampiRequest pointer to the structure into which the values are copied 
 * the information is removed from the internal bookkeeping data
 */
void BK_AMPI_get_AMPI_Request(MPI_Request *request, struct AMPI_Request_S  *ampiRequest);

/**
 * \param request is used as key to look up the associated AMPI_Request_S instance which then is deep copied 
 * \param ampiRequest pointer to the structure into which the values are copied 
 * the information is retained in the internal bookkeeping data
 */
void BK_AMPI_read_AMPI_Request(MPI_Request *request, struct AMPI_Request_S  *ampiRequest);

#endif
