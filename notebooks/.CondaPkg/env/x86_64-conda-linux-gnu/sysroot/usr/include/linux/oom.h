#ifndef __INCLUDE_LINUX_OOM_H
#define __INCLUDE_LINUX_OOM_H

/*
 * /proc/<pid>/oom_adj set to -17 protects from the oom-killer
 */
#define OOM_DISABLE (-17)
/* inclusive */
#define OOM_ADJUST_MIN (-16)
#define OOM_ADJUST_MAX 15

/*
 * /proc/<pid>/oom_score_adj set to OOM_SCORE_ADJ_MIN disables oom killing for
 * pid.
 */
#define OOM_SCORE_ADJ_MIN	(-1000)
#define OOM_SCORE_ADJ_MAX	1000

#endif /* _INCLUDE_LINUX_OOM_H */
