/*
  File autogenerated by gengetopt version 2.23
  generated with the following command:
  gengetopt --output-dir autogen -i args.ggo --default-optional  # --unamed-opts=clusterings

  The developers of gengetopt consider the fixed text that goes in all
  gengetopt output files to be in the public domain:
  we make no copyright claims on it.
*/

/* If we use autoconf.  */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FIX_UNUSED
#define FIX_UNUSED(X) (void) (X) /* avoid warnings for unused params */
#endif

#include <getopt.h>

#include "cmdline.h"

const char *gengetopt_args_info_purpose = "Basic ant tracker (former AntDetect), whose stable functionality is integrated\ninto LAFFTrack/artemis";

const char *gengetopt_args_info_usage = "Usage: atracker [OPTIONS]\n\nExamples:\n  $ ./atracker -fp -kc networks/5K25.cnl tests/5K25_l0.825/5K25_l0.825_796.cnl\n  $ ./atracker -fh -kc -i tests/5K25.cll -ph -l networks/5K25.cnl\ntests/5K25_l0.825/5K25_l0.825_796.cnl\n";

const char *gengetopt_args_info_versiontext = "Author:  (c) Serhii Oleksenko <serhii@lumais.com>, Artem Lutov <lav@lumais.com>\nSources:  https://github.com/LUMAIS/atracker\n";

const char *gengetopt_args_info_description = "Basic ant tracker, which uses YOLO-based ant events detector and several\ntracking techniques to track ant-related objects (e.g., ant, larva, pupa) and\ntheir interaction events (e.g., trophallaxis).\nNOTE: this application is used mainly for internal evaluation and valudation\npurposes before integrating selected functionality into LAFFTrack/artemis.\n";

const char *gengetopt_args_info_help[] = {
  "  -h, --help             Print help and exit",
  "  -V, --version          Print version and exit",
  "  -m, --model=filename   path to the object detector (PyTorch ML model)",
  "  -g, --cuda             computational device for the object detector (CUDA GPU\n                           or CPU}  (default=off)",
  "\n Group: input\n  Input data",
  "  -v, --video=filename   path to the input video",
  "  -i, --img=filename     path to the input image",
  "  -s, --frame_start=INT  start frame index  (default=`0')",
  "  -n, --frame_num=INT    the number of frames  (default=`-1')",
  "  -o, --output=filename  output directory  (default=`.')",
    0
};

typedef enum {ARG_NO
  , ARG_FLAG
  , ARG_STRING
  , ARG_INT
} cmdline_parser_arg_type;

static
void clear_given (struct gengetopt_args_info *args_info);
static
void clear_args (struct gengetopt_args_info *args_info);

static int
cmdline_parser_internal (int argc, char **argv, struct gengetopt_args_info *args_info,
                        struct cmdline_parser_params *params, const char *additional_error);

static int
cmdline_parser_required2 (struct gengetopt_args_info *args_info, const char *prog_name, const char *additional_error);

static char *
gengetopt_strdup (const char *s);

static
void clear_given (struct gengetopt_args_info *args_info)
{
  args_info->help_given = 0 ;
  args_info->version_given = 0 ;
  args_info->model_given = 0 ;
  args_info->cuda_given = 0 ;
  args_info->video_given = 0 ;
  args_info->img_given = 0 ;
  args_info->frame_start_given = 0 ;
  args_info->frame_num_given = 0 ;
  args_info->output_given = 0 ;
  args_info->input_group_counter = 0 ;
}

static
void clear_args (struct gengetopt_args_info *args_info)
{
  FIX_UNUSED (args_info);
  args_info->model_arg = NULL;
  args_info->model_orig = NULL;
  args_info->cuda_flag = 0;
  args_info->video_arg = NULL;
  args_info->video_orig = NULL;
  args_info->img_arg = NULL;
  args_info->img_orig = NULL;
  args_info->frame_start_arg = 0;
  args_info->frame_start_orig = NULL;
  args_info->frame_num_arg = -1;
  args_info->frame_num_orig = NULL;
  args_info->output_arg = gengetopt_strdup (".");
  args_info->output_orig = NULL;
  
}

static
void init_args_info(struct gengetopt_args_info *args_info)
{


  args_info->help_help = gengetopt_args_info_help[0] ;
  args_info->version_help = gengetopt_args_info_help[1] ;
  args_info->model_help = gengetopt_args_info_help[2] ;
  args_info->cuda_help = gengetopt_args_info_help[3] ;
  args_info->video_help = gengetopt_args_info_help[5] ;
  args_info->img_help = gengetopt_args_info_help[6] ;
  args_info->frame_start_help = gengetopt_args_info_help[7] ;
  args_info->frame_num_help = gengetopt_args_info_help[8] ;
  args_info->output_help = gengetopt_args_info_help[9] ;
  
}

void
cmdline_parser_print_version (void)
{
  printf ("%s %s\n",
     (strlen(CMDLINE_PARSER_PACKAGE_NAME) ? CMDLINE_PARSER_PACKAGE_NAME : CMDLINE_PARSER_PACKAGE),
     CMDLINE_PARSER_VERSION);

  if (strlen(gengetopt_args_info_versiontext) > 0)
    printf("\n%s\n", gengetopt_args_info_versiontext);
}

static void print_help_common(void)
{
	size_t len_purpose = strlen(gengetopt_args_info_purpose);
	size_t len_usage = strlen(gengetopt_args_info_usage);

	if (len_usage > 0) {
		printf("%s\n", gengetopt_args_info_usage);
	}
	if (len_purpose > 0) {
		printf("%s\n", gengetopt_args_info_purpose);
	}

	if (len_usage || len_purpose) {
		printf("\n");
	}

	if (strlen(gengetopt_args_info_description) > 0) {
		printf("%s\n\n", gengetopt_args_info_description);
	}
}

void
cmdline_parser_print_help (void)
{
  int i = 0;
  print_help_common();
  while (gengetopt_args_info_help[i])
    printf("%s\n", gengetopt_args_info_help[i++]);
}

void
cmdline_parser_init (struct gengetopt_args_info *args_info)
{
  clear_given (args_info);
  clear_args (args_info);
  init_args_info (args_info);

  args_info->inputs = 0;
  args_info->inputs_num = 0;
}

void
cmdline_parser_params_init(struct cmdline_parser_params *params)
{
  if (params)
    { 
      params->override = 0;
      params->initialize = 1;
      params->check_required = 1;
      params->check_ambiguity = 0;
      params->print_errors = 1;
    }
}

struct cmdline_parser_params *
cmdline_parser_params_create(void)
{
  struct cmdline_parser_params *params = 
    (struct cmdline_parser_params *)malloc(sizeof(struct cmdline_parser_params));
  cmdline_parser_params_init(params);  
  return params;
}

static void
free_string_field (char **s)
{
  if (*s)
    {
      free (*s);
      *s = 0;
    }
}


static void
cmdline_parser_release (struct gengetopt_args_info *args_info)
{
  unsigned int i;
  free_string_field (&(args_info->model_arg));
  free_string_field (&(args_info->model_orig));
  free_string_field (&(args_info->video_arg));
  free_string_field (&(args_info->video_orig));
  free_string_field (&(args_info->img_arg));
  free_string_field (&(args_info->img_orig));
  free_string_field (&(args_info->frame_start_orig));
  free_string_field (&(args_info->frame_num_orig));
  free_string_field (&(args_info->output_arg));
  free_string_field (&(args_info->output_orig));
  
  
  for (i = 0; i < args_info->inputs_num; ++i)
    free (args_info->inputs [i]);

  if (args_info->inputs_num)
    free (args_info->inputs);

  clear_given (args_info);
}


static void
write_into_file(FILE *outfile, const char *opt, const char *arg, const char *values[])
{
  FIX_UNUSED (values);
  if (arg) {
    fprintf(outfile, "%s=\"%s\"\n", opt, arg);
  } else {
    fprintf(outfile, "%s\n", opt);
  }
}


int
cmdline_parser_dump(FILE *outfile, struct gengetopt_args_info *args_info)
{
  int i = 0;

  if (!outfile)
    {
      fprintf (stderr, "%s: cannot dump options to stream\n", CMDLINE_PARSER_PACKAGE);
      return EXIT_FAILURE;
    }

  if (args_info->help_given)
    write_into_file(outfile, "help", 0, 0 );
  if (args_info->version_given)
    write_into_file(outfile, "version", 0, 0 );
  if (args_info->model_given)
    write_into_file(outfile, "model", args_info->model_orig, 0);
  if (args_info->cuda_given)
    write_into_file(outfile, "cuda", 0, 0 );
  if (args_info->video_given)
    write_into_file(outfile, "video", args_info->video_orig, 0);
  if (args_info->img_given)
    write_into_file(outfile, "img", args_info->img_orig, 0);
  if (args_info->frame_start_given)
    write_into_file(outfile, "frame_start", args_info->frame_start_orig, 0);
  if (args_info->frame_num_given)
    write_into_file(outfile, "frame_num", args_info->frame_num_orig, 0);
  if (args_info->output_given)
    write_into_file(outfile, "output", args_info->output_orig, 0);
  

  i = EXIT_SUCCESS;
  return i;
}

int
cmdline_parser_file_save(const char *filename, struct gengetopt_args_info *args_info)
{
  FILE *outfile;
  int i = 0;

  outfile = fopen(filename, "w");

  if (!outfile)
    {
      fprintf (stderr, "%s: cannot open file for writing: %s\n", CMDLINE_PARSER_PACKAGE, filename);
      return EXIT_FAILURE;
    }

  i = cmdline_parser_dump(outfile, args_info);
  fclose (outfile);

  return i;
}

void
cmdline_parser_free (struct gengetopt_args_info *args_info)
{
  cmdline_parser_release (args_info);
}

/** @brief replacement of strdup, which is not standard */
char *
gengetopt_strdup (const char *s)
{
  char *result = 0;
  if (!s)
    return result;

  result = (char*)malloc(strlen(s) + 1);
  if (result == (char*)0)
    return (char*)0;
  strcpy(result, s);
  return result;
}

static void
reset_group_input(struct gengetopt_args_info *args_info)
{
  if (! args_info->input_group_counter)
    return;
  
  args_info->video_given = 0 ;
  free_string_field (&(args_info->video_arg));
  free_string_field (&(args_info->video_orig));
  args_info->img_given = 0 ;
  free_string_field (&(args_info->img_arg));
  free_string_field (&(args_info->img_orig));

  args_info->input_group_counter = 0;
}

int
cmdline_parser (int argc, char **argv, struct gengetopt_args_info *args_info)
{
  return cmdline_parser2 (argc, argv, args_info, 0, 1, 1);
}

int
cmdline_parser_ext (int argc, char **argv, struct gengetopt_args_info *args_info,
                   struct cmdline_parser_params *params)
{
  int result;
  result = cmdline_parser_internal (argc, argv, args_info, params, 0);

  if (result == EXIT_FAILURE)
    {
      cmdline_parser_free (args_info);
      exit (EXIT_FAILURE);
    }
  
  return result;
}

int
cmdline_parser2 (int argc, char **argv, struct gengetopt_args_info *args_info, int override, int initialize, int check_required)
{
  int result;
  struct cmdline_parser_params params;
  
  params.override = override;
  params.initialize = initialize;
  params.check_required = check_required;
  params.check_ambiguity = 0;
  params.print_errors = 1;

  result = cmdline_parser_internal (argc, argv, args_info, &params, 0);

  if (result == EXIT_FAILURE)
    {
      cmdline_parser_free (args_info);
      exit (EXIT_FAILURE);
    }
  
  return result;
}

int
cmdline_parser_required (struct gengetopt_args_info *args_info, const char *prog_name)
{
  int result = EXIT_SUCCESS;

  if (cmdline_parser_required2(args_info, prog_name, 0) > 0)
    result = EXIT_FAILURE;

  if (result == EXIT_FAILURE)
    {
      cmdline_parser_free (args_info);
      exit (EXIT_FAILURE);
    }
  
  return result;
}

int
cmdline_parser_required2 (struct gengetopt_args_info *args_info, const char *prog_name, const char *additional_error)
{
  int error_occurred = 0;
  FIX_UNUSED (additional_error);

  /* checks for required options */
  if (args_info->input_group_counter == 0)
    {
      fprintf (stderr, "%s: %d options of group input were given. One is required%s.\n", prog_name, args_info->input_group_counter, (additional_error ? additional_error : ""));
      error_occurred = 1;
    }
  

  /* checks for dependences among options */
  if (args_info->frame_start_given && ! args_info->video_given)
    {
      fprintf (stderr, "%s: '--frame_start' ('-s') option depends on option 'video'%s\n", prog_name, (additional_error ? additional_error : ""));
      error_occurred = 1;
    }
  if (args_info->frame_num_given && ! args_info->video_given)
    {
      fprintf (stderr, "%s: '--frame_num' ('-n') option depends on option 'video'%s\n", prog_name, (additional_error ? additional_error : ""));
      error_occurred = 1;
    }

  return error_occurred;
}


static char *package_name = 0;

/**
 * @brief updates an option
 * @param field the generic pointer to the field to update
 * @param orig_field the pointer to the orig field
 * @param field_given the pointer to the number of occurrence of this option
 * @param prev_given the pointer to the number of occurrence already seen
 * @param value the argument for this option (if null no arg was specified)
 * @param possible_values the possible values for this option (if specified)
 * @param default_value the default value (in case the option only accepts fixed values)
 * @param arg_type the type of this option
 * @param check_ambiguity @see cmdline_parser_params.check_ambiguity
 * @param override @see cmdline_parser_params.override
 * @param no_free whether to free a possible previous value
 * @param multiple_option whether this is a multiple option
 * @param long_opt the corresponding long option
 * @param short_opt the corresponding short option (or '-' if none)
 * @param additional_error possible further error specification
 */
static
int update_arg(void *field, char **orig_field,
               unsigned int *field_given, unsigned int *prev_given, 
               char *value, const char *possible_values[],
               const char *default_value,
               cmdline_parser_arg_type arg_type,
               int check_ambiguity, int override,
               int no_free, int multiple_option,
               const char *long_opt, char short_opt,
               const char *additional_error)
{
  char *stop_char = 0;
  const char *val = value;
  int found;
  char **string_field;
  FIX_UNUSED (field);

  stop_char = 0;
  found = 0;

  if (!multiple_option && prev_given && (*prev_given || (check_ambiguity && *field_given)))
    {
      if (short_opt != '-')
        fprintf (stderr, "%s: `--%s' (`-%c') option given more than once%s\n", 
               package_name, long_opt, short_opt,
               (additional_error ? additional_error : ""));
      else
        fprintf (stderr, "%s: `--%s' option given more than once%s\n", 
               package_name, long_opt,
               (additional_error ? additional_error : ""));
      return 1; /* failure */
    }

  FIX_UNUSED (default_value);
    
  if (field_given && *field_given && ! override)
    return 0;
  if (prev_given)
    (*prev_given)++;
  if (field_given)
    (*field_given)++;
  if (possible_values)
    val = possible_values[found];

  switch(arg_type) {
  case ARG_FLAG:
    *((int *)field) = !*((int *)field);
    break;
  case ARG_INT:
    if (val) *((int *)field) = strtol (val, &stop_char, 0);
    break;
  case ARG_STRING:
    if (val) {
      string_field = (char **)field;
      if (!no_free && *string_field)
        free (*string_field); /* free previous string */
      *string_field = gengetopt_strdup (val);
    }
    break;
  default:
    break;
  };

  /* check numeric conversion */
  switch(arg_type) {
  case ARG_INT:
    if (val && !(stop_char && *stop_char == '\0')) {
      fprintf(stderr, "%s: invalid numeric value: %s\n", package_name, val);
      return 1; /* failure */
    }
    break;
  default:
    ;
  };

  /* store the original value */
  switch(arg_type) {
  case ARG_NO:
  case ARG_FLAG:
    break;
  default:
    if (value && orig_field) {
      if (no_free) {
        *orig_field = value;
      } else {
        if (*orig_field)
          free (*orig_field); /* free previous string */
        *orig_field = gengetopt_strdup (value);
      }
    }
  };

  return 0; /* OK */
}


int
cmdline_parser_internal (
  int argc, char **argv, struct gengetopt_args_info *args_info,
                        struct cmdline_parser_params *params, const char *additional_error)
{
  int c;	/* Character of the parsed option.  */

  int error_occurred = 0;
  struct gengetopt_args_info local_args_info;
  
  int override;
  int initialize;
  int check_required;
  int check_ambiguity;
  
  package_name = argv[0];
  
  /* TODO: Why is this here? It is not used anywhere. */
  override = params->override;
  FIX_UNUSED(override);

  initialize = params->initialize;
  check_required = params->check_required;

  /* TODO: Why is this here? It is not used anywhere. */
  check_ambiguity = params->check_ambiguity;
  FIX_UNUSED(check_ambiguity);

  if (initialize)
    cmdline_parser_init (args_info);

  cmdline_parser_init (&local_args_info);

  optarg = 0;
  optind = 0;
  opterr = params->print_errors;
  optopt = '?';

  while (1)
    {
      int option_index = 0;

      static struct option long_options[] = {
        { "help",	0, NULL, 'h' },
        { "version",	0, NULL, 'V' },
        { "model",	1, NULL, 'm' },
        { "cuda",	0, NULL, 'g' },
        { "video",	1, NULL, 'v' },
        { "img",	1, NULL, 'i' },
        { "frame_start",	1, NULL, 's' },
        { "frame_num",	1, NULL, 'n' },
        { "output",	1, NULL, 'o' },
        { 0,  0, 0, 0 }
      };

      c = getopt_long (argc, argv, "hVm:gv:i:s:n:o:", long_options, &option_index);

      if (c == -1) break;	/* Exit from `while (1)' loop.  */

      switch (c)
        {
        case 'h':	/* Print help and exit.  */
          cmdline_parser_print_help ();
          cmdline_parser_free (&local_args_info);
          exit (EXIT_SUCCESS);

        case 'V':	/* Print version and exit.  */
          cmdline_parser_print_version ();
          cmdline_parser_free (&local_args_info);
          exit (EXIT_SUCCESS);

        case 'm':	/* path to the object detector (PyTorch ML model).  */
        
        
          if (update_arg( (void *)&(args_info->model_arg), 
               &(args_info->model_orig), &(args_info->model_given),
              &(local_args_info.model_given), optarg, 0, 0, ARG_STRING,
              check_ambiguity, override, 0, 0,
              "model", 'm',
              additional_error))
            goto failure;
        
          break;
        case 'g':	/* computational device for the object detector (CUDA GPU or CPU}.  */
        
        
          if (update_arg((void *)&(args_info->cuda_flag), 0, &(args_info->cuda_given),
              &(local_args_info.cuda_given), optarg, 0, 0, ARG_FLAG,
              check_ambiguity, override, 1, 0, "cuda", 'g',
              additional_error))
            goto failure;
        
          break;
        case 'v':	/* path to the input video.  */
        
          if (args_info->input_group_counter && override)
            reset_group_input (args_info);
          args_info->input_group_counter += 1;
        
          if (update_arg( (void *)&(args_info->video_arg), 
               &(args_info->video_orig), &(args_info->video_given),
              &(local_args_info.video_given), optarg, 0, 0, ARG_STRING,
              check_ambiguity, override, 0, 0,
              "video", 'v',
              additional_error))
            goto failure;
        
          break;
        case 'i':	/* path to the input image.  */
        
          if (args_info->input_group_counter && override)
            reset_group_input (args_info);
          args_info->input_group_counter += 1;
        
          if (update_arg( (void *)&(args_info->img_arg), 
               &(args_info->img_orig), &(args_info->img_given),
              &(local_args_info.img_given), optarg, 0, 0, ARG_STRING,
              check_ambiguity, override, 0, 0,
              "img", 'i',
              additional_error))
            goto failure;
        
          break;
        case 's':	/* start frame index.  */
        
        
          if (update_arg( (void *)&(args_info->frame_start_arg), 
               &(args_info->frame_start_orig), &(args_info->frame_start_given),
              &(local_args_info.frame_start_given), optarg, 0, "0", ARG_INT,
              check_ambiguity, override, 0, 0,
              "frame_start", 's',
              additional_error))
            goto failure;
        
          break;
        case 'n':	/* the number of frames.  */
        
        
          if (update_arg( (void *)&(args_info->frame_num_arg), 
               &(args_info->frame_num_orig), &(args_info->frame_num_given),
              &(local_args_info.frame_num_given), optarg, 0, "-1", ARG_INT,
              check_ambiguity, override, 0, 0,
              "frame_num", 'n',
              additional_error))
            goto failure;
        
          break;
        case 'o':	/* output directory.  */
        
        
          if (update_arg( (void *)&(args_info->output_arg), 
               &(args_info->output_orig), &(args_info->output_given),
              &(local_args_info.output_given), optarg, 0, ".", ARG_STRING,
              check_ambiguity, override, 0, 0,
              "output", 'o',
              additional_error))
            goto failure;
        
          break;

        case 0:	/* Long option with no short option */
        case '?':	/* Invalid option.  */
          /* `getopt_long' already printed an error message.  */
          goto failure;

        default:	/* bug: option not considered.  */
          fprintf (stderr, "%s: option unknown: %c%s\n", CMDLINE_PARSER_PACKAGE, c, (additional_error ? additional_error : ""));
          abort ();
        } /* switch */
    } /* while */

  if (args_info->input_group_counter > 1)
    {
      fprintf (stderr, "%s: %d options of group input were given. One is required%s.\n", argv[0], args_info->input_group_counter, (additional_error ? additional_error : ""));
      error_occurred = 1;
    }
  


  if (check_required)
    {
      error_occurred += cmdline_parser_required2 (args_info, argv[0], additional_error);
    }

  cmdline_parser_release (&local_args_info);

  if ( error_occurred )
    return (EXIT_FAILURE);

  if (optind < argc)
    {
      int i = 0 ;
      int found_prog_name = 0;
      /* whether program name, i.e., argv[0], is in the remaining args
         (this may happen with some implementations of getopt,
          but surely not with the one included by gengetopt) */

      i = optind;
      while (i < argc)
        if (argv[i++] == argv[0]) {
          found_prog_name = 1;
          break;
        }
      i = 0;

      args_info->inputs_num = argc - optind - found_prog_name;
      args_info->inputs =
        (char **)(malloc ((args_info->inputs_num)*sizeof(char *))) ;
      while (optind < argc)
        if (argv[optind++] != argv[0])
          args_info->inputs[ i++ ] = gengetopt_strdup (argv[optind-1]) ;
    }

  return 0;

failure:
  
  cmdline_parser_release (&local_args_info);
  return (EXIT_FAILURE);
}
/* vim: set ft=c noet ts=8 sts=8 sw=8 tw=80 nojs spell : */
